import os
import sys
import json
import argparse
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# =====================================================================
# 0. 模型定义 / 预处理：优先复用 train_pipeline_v2.py，没有就退回备份定义
# =====================================================================

_TRAIN_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _TRAIN_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_SCRIPT_DIR)

try:
    from train_pipeline_v2 import (
        RadicalAwareSwinModel,
        preprocess_bgr_image,
        PAD_OR_UNK,
    )
    _REUSED_TRAINING_CODE = True
except ImportError:
    _REUSED_TRAINING_CODE = False
    print(
        "[警告] 没有在同目录下找到 train_pipeline_v2.py，使用本脚本内置的备份定义。\n"
        "       这份备份必须和训练脚本里的模型结构/预处理逐字保持一致——历史上已经\n"
        "       因为两边各自维护、悄悄长歪而出过至少两次真实 bug。强烈建议把\n"
        "       train_pipeline_v2.py 拷到这个推理脚本同一目录下，让下面这段\n"
        "       import 直接生效，而不是依赖这份备份。"
    )

    PAD_OR_UNK = "<PAD_OR_UNK>"
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess_bgr_image(img_bgr: np.ndarray, target_size: int,
                              mask: Optional[np.ndarray] = None) -> np.ndarray:
        img = cv2.resize(img_bgr, (target_size, target_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))
        if mask is not None:
            m = cv2.resize(mask, (target_size, target_size))
            m = (m.astype(np.float32) / 255.0)[None, :, :]
            img = np.concatenate([img, m], axis=0)
        return img

    class RadicalAwareSwinModel(nn.Module):
        def __init__(self, num_classes: int, num_radicals: int = 0,
                     swin_variant: str = "swin_t", pretrained: bool = False,
                     extra_channels: int = 0):
            super().__init__()
            import torchvision.models as tvm
            if swin_variant == "swin_t":
                swin = tvm.swin_t(weights=None)
            elif swin_variant == "swin_v2_t":
                swin = tvm.swin_v2_t(weights=None)
            else:
                raise ValueError(f"未知 swin_variant: {swin_variant}")

            if extra_channels > 0:
                old_conv = swin.features[0][0]
                new_conv = nn.Conv2d(
                    old_conv.in_channels + extra_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                    padding=old_conv.padding, bias=(old_conv.bias is not None),
                )
                with torch.no_grad():
                    new_conv.weight[:, :old_conv.in_channels] = old_conv.weight
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                    for c in range(extra_channels):
                        new_conv.weight[:, old_conv.in_channels + c: old_conv.in_channels + c + 1] = mean_w
                    if old_conv.bias is not None:
                        new_conv.bias[:] = old_conv.bias
                swin.features[0][0] = new_conv

            self.features = swin.features
            self.norm = swin.norm
            self.use_radical_head = num_radicals > 0
            self.head = nn.Linear(768, num_classes)
            if self.use_radical_head:
                self.radical_head = nn.Linear(768, num_radicals)

        def forward(self, x):
            feat = self.features(x)
            feat = self.norm(feat)
            feat = feat.permute(0, 3, 1, 2)
            feat = nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            feat = torch.flatten(feat, 1)
            cls_logits = self.head(feat)
            radical_logits = self.radical_head(feat) if self.use_radical_head else None
            return cls_logits, radical_logits


# =====================================================================
# 1. 语言模型 Beam Search 解码器（沿用上一版逻辑，只修 decode() 的返回类型）
# =====================================================================

class UltimateBeamSearchDecoderV10:
    def __init__(self, assets_dir,
                 alpha=1.0, beta=0.25, gamma=0.45, theta=0.25, delta=0.30,
                 ppmi_scale=5.0, eta=0.0,
                 ppmi_lambda=(0.5, 0.3, 0.2)):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.delta = delta
        self.eta = eta
        self.ppmi_scale = ppmi_scale
        self.ppmi_lambda = ppmi_lambda

        with open(os.path.join(assets_dir, "unigram_counts.pkl"), 'rb') as f:
            self.unigram_counts = pickle.load(f)
        with open(os.path.join(assets_dir, "bigram_counts.pkl"), 'rb') as f:
            self.bigram_counts = pickle.load(f)
        with open(os.path.join(assets_dir, "trigram_counts.pkl"), 'rb') as f:
            self.trigram_counts = pickle.load(f)
        with open(os.path.join(assets_dir, "ppmi_1.pkl"), 'rb') as f:
            self.ppmi_1 = pickle.load(f)
        with open(os.path.join(assets_dir, "ppmi_2.pkl"), 'rb') as f:
            self.ppmi_2 = pickle.load(f)
        with open(os.path.join(assets_dir, "ppmi_3.pkl"), 'rb') as f:
            self.ppmi_3 = pickle.load(f)
        self.ppmi_dicts = {1: self.ppmi_1, 2: self.ppmi_2, 3: self.ppmi_3}

        with open(os.path.join(assets_dir, "artifact_prior.pkl"), 'rb') as f:
            self.artifact_prior = pickle.load(f)
        with open(os.path.join(assets_dir, "metadata.pkl"), 'rb') as f:
            self.metadata = pickle.load(f)
        with open(os.path.join(assets_dir, "kn_stats.pkl"), 'rb') as f:
            self.kn_stats = pickle.load(f)

        self.vocab_size = len(self.unigram_counts)
        self.total_tokens = sum(self.unigram_counts.values())
        self.epsilon = 1e-8

        self.ppmi_stats = {}
        for off in [1, 2, 3]:
            self.ppmi_stats[off] = {
                "mean": self.metadata[f"ppmi_{off}_mean"],
                "std": self.metadata[f"ppmi_{off}_std"],
            }
            if self.ppmi_stats[off]["std"] < 1e-8:
                self.ppmi_stats[off]["std"] = 1.0

        self.bigram_continuation = self.kn_stats["bigram_continuation"]
        self.bigram_successor = self.kn_stats["bigram_successor"]
        self.trigram_successor = self.kn_stats["trigram_successor"]
        self.trigram_context_counts = self.kn_stats["trigram_context_counts"]
        self.bigram_context_counts = self.kn_stats["bigram_context_counts"]
        self.bigram_n1 = self.kn_stats["bigram_n1"]
        self.bigram_n2 = self.kn_stats["bigram_n2"]
        self.bigram_n3p = self.kn_stats["bigram_n3p"]
        self.trigram_n1 = self.kn_stats["trigram_n1"]
        self.trigram_n2 = self.kn_stats["trigram_n2"]
        self.trigram_n3p = self.kn_stats["trigram_n3p"]

        self.D1_b, self.D2_b, self.D3_b = self._compute_mkn_discounts(self.kn_stats["freq_of_freq_bigram"])
        self.D1_t, self.D2_t, self.D3_t = self._compute_mkn_discounts(self.kn_stats["freq_of_freq_trigram"])

        self.unigram_probs = {ch: cnt / self.total_tokens for ch, cnt in self.unigram_counts.items()}

    def _compute_mkn_discounts(self, freq_of_freq):
        n1 = freq_of_freq.get(1, 0)
        n2 = freq_of_freq.get(2, 0)
        n3 = freq_of_freq.get(3, 0)
        n4 = freq_of_freq.get(4, 0)
        if n1 == 0:
            return 1.0, 2.0, 3.0
        Y = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) > 0 else 0.0
        D1 = 1 - 2 * Y * (n2 / n1) if n1 > 0 else 1.0
        D2 = 2 - 3 * Y * (n3 / n2) if n2 > 0 else 2.0
        D3 = 3 - 4 * Y * (n4 / n3) if n3 > 0 else 3.0
        return max(0.1, D1), max(0.1, D2), max(0.1, D3)

    def _get_discount(self, count, is_bigram=True):
        if is_bigram:
            if count == 1:
                return self.D1_b
            if count == 2:
                return self.D2_b
            return self.D3_b
        else:
            if count == 1:
                return self.D1_t
            if count == 2:
                return self.D2_t
            return self.D3_t

    def _get_ppmi_logprob(self, w1, w2, offset):
        if offset not in self.ppmi_dicts:
            return np.log(self.epsilon)
        raw = self.ppmi_dicts[offset].get(w1, {}).get(w2, 0.0)
        if raw == 0.0:
            return np.log(self.epsilon)
        mean = self.ppmi_stats[offset]["mean"]
        std = self.ppmi_stats[offset]["std"]
        z = (raw - mean) / std
        log_sigmoid = -np.log(1 + np.exp(-z))
        return self.ppmi_scale * log_sigmoid

    def _get_aggregated_ppmi(self, seq, cand_char):
        max_k = min(3, len(seq))
        total = 0.0
        for k, lam in zip(range(1, max_k + 1), self.ppmi_lambda):
            if k <= len(seq):
                score = self._get_ppmi_logprob(seq[-k], cand_char, offset=k)
                total += lam * score
        if len(seq) >= 3:
            p3 = self._get_ppmi_logprob(seq[-3], cand_char, offset=3)
            p2 = self._get_ppmi_logprob(seq[-2], cand_char, offset=2)
            if p3 < -10:
                p1 = self._get_ppmi_logprob(seq[-1], cand_char, offset=1) if len(seq) >= 1 else np.log(self.epsilon)
                p3_smooth = 0.6 * p2 + 0.4 * p3
                total = self.ppmi_lambda[0] * p1 + self.ppmi_lambda[1] * p2 + self.ppmi_lambda[2] * p3_smooth
        return total

    def _kn_bigram_prob(self, w1, w2):
        cont_w2 = self.bigram_continuation.get(w2, 0)
        total_cont = sum(self.bigram_continuation.values())
        cont_prob = (cont_w2 / total_cont) if total_cont > 0 else (1.0 / self.vocab_size)

        count = self.bigram_counts.get(w1, {}).get(w2, 0)
        context = self.bigram_context_counts.get(w1, 0)
        if context > 0:
            n1 = self.bigram_n1.get(w1, 0)
            n2 = self.bigram_n2.get(w1, 0)
            n3p = self.bigram_n3p.get(w1, 0)
            D = self._get_discount(count, is_bigram=True)
            disc = max(count - D, 0) / context
            lambda_ = (self.D1_b * n1 + self.D2_b * n2 + self.D3_b * n3p) / context
            lambda_ = min(lambda_, 1.0)
            prob = disc + lambda_ * cont_prob
        else:
            prob = cont_prob
        return max(prob, self.epsilon)

    def _kn_trigram_prob(self, w1, w2, w3):
        key = f"{w1}_{w2}"
        count_w1_w2_w3 = self.trigram_counts.get(key, {}).get(w3, 0)
        context = self.trigram_context_counts.get(key, 0)
        if context > 0:
            n1 = self.trigram_n1.get(key, 0)
            n2 = self.trigram_n2.get(key, 0)
            n3p = self.trigram_n3p.get(key, 0)
            D = self._get_discount(count_w1_w2_w3, is_bigram=False)
            disc = max(count_w1_w2_w3 - D, 0) / context
            lambda_ = (self.D1_t * n1 + self.D2_t * n2 + self.D3_t * n3p) / context
            lambda_ = min(lambda_, 1.0)
            backoff = self._kn_bigram_prob(w2, w3)
            prob = disc + lambda_ * backoff
        else:
            prob = self._kn_bigram_prob(w2, w3)
        return max(prob, self.epsilon)

    def _get_mixture_prior_score(self, visual_distribution, cand_char):
        total_vis = sum(p for _, p in visual_distribution)
        if total_vis == 0:
            return np.log(self.epsilon)
        mixture_prob = 0.0
        for pred_char, vis_prob in visual_distribution:
            norm_prob = vis_prob / total_vis
            gt_prob = self.artifact_prior.get(pred_char, {}).get(cand_char, self.epsilon)
            mixture_prob += norm_prob * gt_prob
        mixture_prob = max(mixture_prob, self.epsilon)
        if self.eta > 0:
            unigram_prob = self.unigram_probs.get(cand_char, self.epsilon)
            final_prob = mixture_prob * (unigram_prob ** self.eta)
        else:
            final_prob = mixture_prob
        return np.log(max(final_prob, self.epsilon))

    def decode(self, all_candidates, beam_size=10) -> List[str]:
        """
        [关键修复] 返回 list[str]（每个元素是恰好一个 token，对应一个 bbox），
        不再在内部 "".join() 压成字符串。占位符 PAD_OR_UNK 这种多字符 token
        在 list 里依然是"一个元素"，不会被拆散成单个字符。
        """
        if not all_candidates:
            return []

        first_step = all_candidates[0]
        beams = []
        for char, prob in first_step:
            visual_score = np.log(max(prob, self.epsilon))
            beams.append({"seq": [char], "score": self.alpha * visual_score})

        for t in range(1, len(all_candidates)):
            current_candidates = all_candidates[t]
            visual_dist = current_candidates

            new_beams = []
            for beam in beams:
                seq = beam["seq"]
                old_score = beam["score"]
                prev_char = seq[-1]
                prev2_char = seq[-2] if len(seq) >= 2 else None

                for cand_char, cand_prob in current_candidates:
                    visual_score = np.log(max(cand_prob, self.epsilon))
                    mixture_score = self._get_mixture_prior_score(visual_dist, cand_char)
                    ppmi_sum = self._get_aggregated_ppmi(seq, cand_char)

                    if prev2_char:
                        trigram_log = np.log(self._kn_trigram_prob(prev2_char, prev_char, cand_char))
                    else:
                        trigram_log = np.log(self.epsilon)
                    bigram_log = np.log(self._kn_bigram_prob(prev_char, cand_char))

                    step_score = (
                        (self.alpha * visual_score)
                        + (self.beta * ppmi_sum)
                        + (self.gamma * trigram_log)
                        + (self.theta * bigram_log)
                        + (self.delta * mixture_score)
                    )
                    new_beams.append({"seq": seq + [cand_char], "score": old_score + step_score})

            beams = sorted(new_beams, key=lambda x: x["score"], reverse=True)[:beam_size]

        best_beam = max(beams, key=lambda x: x["score"])
        return best_beam["seq"]  # list[str]，不再 join


def greedy_visual_decode(all_candidates) -> List[str]:
    """
    [对照实验用] 纯视觉 top-1，不经过任何语言模型——每个框直接取分类器自己
    给出的最高置信度候选。用来验证 beam search 里那套 n-gram/PPMI/混合先验
    到底是不是真的在帮忙：跑一遍这个、跑一遍正常 beam search，本地对比 F1。
    """
    return [max(cands, key=lambda x: x[1])[0] for cands in all_candidates] if all_candidates else []


# =====================================================================
# 2. 端到端 OCR 推理引擎
# =====================================================================

class AntiqueOCRSystemV11:
    def __init__(self, obb_model_path, classifier_model_path, assets_dir,
                 device='cuda', topk=5, beam_size=10,
                 use_language_model=True, use_aspect_padding=False):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.topk = topk
        self.beam_size = beam_size
        self.use_language_model = use_language_model
        # [训练/推理预处理对齐] 当前训好的模型，训练时是直接 resize（会有长宽比
        # 畸变），所以这里默认也直接 resize，不做 padding-to-square。如果以后
        # 在 train_pipeline_v2.py 的 preprocess_bgr_image 里加上了 padding 并
        # 重新训练，再把这里改成 True，两边才能继续对齐。
        self.use_aspect_padding = use_aspect_padding

        print(f"🔄 正在读取 Checkpoint: {classifier_model_path}")
        checkpoint = torch.load(classifier_model_path, map_location=self.device)

        self.char2id = checkpoint.get('char2id_vocab') or checkpoint.get('char2id')
        if self.char2id is None:
            raise RuntimeError(
                "Checkpoint 中没有找到 'char2id_vocab'（或 'char2id'）字段，"
                "确认一下这个 .pt 文件是不是 train_pipeline_v2.py 保存出来的。"
            )
        self.id2char = {idx: char for char, idx in self.char2id.items()}
        self.num_classes = len(self.char2id)
        self.unk_token = PAD_OR_UNK if PAD_OR_UNK in self.char2id else (
            "PAD_OR_UNK" if "PAD_OR_UNK" in self.char2id else PAD_OR_UNK
        )
        print(f"✅ 词表加载完成，分类头类别数: {self.num_classes}")

        flags = checkpoint.get('flags', {})
        swin_variant = flags.get("swin_variant", "swin_t")
        extra_channels = 1 if flags.get("use_mask_channel", False) else 0

        radical_vocab = checkpoint.get("radical_vocab", None)
        num_radicals = len(radical_vocab) if radical_vocab else 0

        self.model = RadicalAwareSwinModel(
            num_classes=self.num_classes,
            num_radicals=num_radicals,
            swin_variant=swin_variant,
            extra_channels=extra_channels,
        )

        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # accelerator.unwrap_model() 保存出来的 state_dict 本身不会带 "module."
        # 前缀，这里留着只是防御性兜底（万一哪天换了别的保存方式），不影响当前
        # 实际使用的 checkpoint 格式。
        clean_state = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        self.model.load_state_dict(clean_state, strict=True)
        self.model.to(self.device).eval()
        print("🏆 模型权重已严格对齐加载（strict=True）。")

        self.detector = YOLO(obb_model_path)

        self.decoder = None
        if self.use_language_model:
            self.decoder = UltimateBeamSearchDecoderV10(assets_dir=assets_dir)
        else:
            print("ℹ️ use_language_model=False：跳过语言模型资产加载，"
                  "推理时每个框直接用分类器 top-1 结果，不经过 beam search。")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.image_size = flags.get("image_size", 128)

    def _predict_candidates(self, crop_img) -> List[Tuple[str, float]]:
        if crop_img is None or crop_img.size == 0:
            # 这一步是唯一候选（没有其它候选跟它竞争），prob=1.0 只是一个占位
            # 置信度，不代表真实预测把握，不会因此扭曲其它正常预测的相对排序。
            return [(self.unk_token, 1.0)]

        if self.use_aspect_padding:
            h, w = crop_img.shape[:2]
            side = max(h, w)
            pad_img = np.zeros((side, side, 3), dtype=np.uint8)
            dx, dy = (side - w) // 2, (side - h) // 2
            pad_img[dy:dy + h, dx:dx + w] = crop_img
            src_img = pad_img
        else:
            # 和当前训好的模型保持一致：直接 resize，不做等比例 padding。
            src_img = crop_img

        crop_chw = preprocess_bgr_image(src_img, self.image_size)
        tensor = torch.from_numpy(crop_chw).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        topk_idx = np.argsort(probs)[-self.topk:][::-1]
        candidates = [(self.id2char.get(int(i), self.unk_token), float(probs[i])) for i in topk_idx]
        return candidates

    def _sort_boxes(self, obb_data) -> List[dict]:
        """
        旋转框按"自右向左、自上而下"的古籍阅读顺序排序。
        列宽阈值改成按本图框宽度中位数自适应（之前是写死 60px，不同分辨率/
        字号的图会在列边界把同一列的字切错列，进而污染依赖阅读顺序打分的
        bigram/trigram 部分）。
        """
        boxes = []
        for corners in obb_data:
            x_min, x_max = corners[:, 0].min(), corners[:, 0].max()
            y_min, y_max = corners[:, 1].min(), corners[:, 1].max()
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
            boxes.append({'cx': cx, 'cy': cy, 'w': x_max - x_min, 'rect': (x_min, y_min, x_max, y_max)})

        if len(boxes) <= 1:
            return boxes

        median_w = float(np.median([b['w'] for b in boxes]))
        col_threshold = max(median_w * 1.5, 1.0)  # 防止 median_w 异常小把阈值压成 0

        cols = defaultdict(list)
        for b in boxes:
            col_id = int(b['cx'] // col_threshold)
            cols[col_id].append(b)

        sorted_cols = sorted(cols.items(), key=lambda x: -x[0])  # 列从右到左
        sorted_boxes = []
        for _, col_items in sorted_cols:
            col_items.sort(key=lambda x: x['cy'])  # 列内从上到下
            sorted_boxes.extend(col_items)
        return sorted_boxes

    def run(self, image_path: str, conf: float = 0.5, iou: float = 0.3) -> List[dict]:
        full_img = cv2.imread(image_path)
        if full_img is None:
            return []

        results = self.detector.predict(image_path, conf=conf, iou=iou, verbose=False)[0]
        if results.obb is None or len(results.obb) == 0:
            return []

        obb_data = results.obb.xyxyxyxy.cpu().numpy()
        sorted_boxes = self._sort_boxes(obb_data)

        all_candidates: List[List[Tuple[str, float]]] = []
        bboxes: List[List[int]] = []
        for b in sorted_boxes:
            x_min, y_min, x_max, y_max = b['rect']
            pad = 5
            x1 = max(0, int(x_min - pad))
            y1 = max(0, int(y_min - pad))
            x2 = min(full_img.shape[1], int(x_max + pad))
            y2 = min(full_img.shape[0], int(y_max + pad))

            crop = full_img[y1:y2, x1:x2]
            all_candidates.append(self._predict_candidates(crop))
            bboxes.append([int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)])

        if self.use_language_model:
            decoded_tokens = self.decoder.decode(all_candidates, beam_size=self.beam_size)
        else:
            decoded_tokens = greedy_visual_decode(all_candidates)

        # [关键修复] decoded_tokens 现在始终是 list[str]，每个元素正好对应一个
        # bbox，长度对不齐时按 token（不是按字符）补齐/截断，不会再把多字符
        # 占位符拆散成单个英文字母污染后面的框。
        if len(decoded_tokens) < len(bboxes):
            decoded_tokens = decoded_tokens + [self.unk_token] * (len(bboxes) - len(decoded_tokens))
        elif len(decoded_tokens) > len(bboxes):
            decoded_tokens = decoded_tokens[:len(bboxes)]

        return [{"bbox": bbox, "text": tok} for bbox, tok in zip(bboxes, decoded_tokens)]


# =====================================================================
# 3. 推理控制中心
# =====================================================================

def get_default_config() -> dict:
    return {
        "OBB_MODEL": "/app/models/last.pt",
        "CLASSIFIER_MODEL": "/app/models/swin_recognizer_best.pt",
        "ASSETS_DIR": "/app/models/",
        "INPUT_DIR": "/saisdata/50/eval/images/",
        "OUTPUT_JSON": "/saisresult/prediction.json",
        "DEVICE": "cuda",
        "TOPK": 5,
        "BEAM_SIZE": 10,
        "CONF": 0.5,
        "IOU": 0.3,
        "use_language_model": True,    # False = 纯视觉 top-1 对照实验，不需要 ASSETS_DIR
        "use_aspect_padding": False,   # 和当前训好的模型保持一致；除非重训过带 padding 的版本，不要改
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="可选：json 文件覆盖默认配置")
    args = parser.parse_args()

    config = get_default_config()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config.update(json.load(f))

    print("🚀 开始初始化端到端古文字识别推理引擎...")
    ocr = AntiqueOCRSystemV11(
        obb_model_path=config["OBB_MODEL"],
        classifier_model_path=config["CLASSIFIER_MODEL"],
        assets_dir=config["ASSETS_DIR"],
        device=config["DEVICE"],
        topk=config["TOPK"],
        beam_size=config["BEAM_SIZE"],
        use_language_model=config["use_language_model"],
        use_aspect_padding=config["use_aspect_padding"],
    )

    image_files = [f for f in os.listdir(config["INPUT_DIR"]) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    print(f"📈 扫描完毕，待评估目标测试集共 {len(image_files)} 张图片。")

    final_predictions = {}
    for idx, img_name in enumerate(image_files):
        img_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(config["INPUT_DIR"], img_name)
        try:
            result = ocr.run(img_path, conf=config["CONF"], iou=config["IOU"])
            final_predictions[img_id] = result
            if (idx + 1) % 50 == 0:
                print(f"☕ 已完成进度: {idx + 1}/{len(image_files)}")
        except Exception as e:
            print(f"❌ 处理单张图片异常 [{img_name}]: {e}")
            final_predictions[img_id] = []

    os.makedirs(os.path.dirname(config["OUTPUT_JSON"]), exist_ok=True)
    with open(config["OUTPUT_JSON"], 'w', encoding='utf-8') as f:
        json.dump(final_predictions, f, ensure_ascii=False, indent=2)
    print(f"🎯 推理结果已写入: {config['OUTPUT_JSON']}")


if __name__ == "__main__":
    main()