%%writefile test_12.py
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import json
import pickle
from collections import defaultdict
from scipy.spatial import KDTree
from ultralytics import YOLO
import torchvision.models as models

# ==========================================
# 1. 分类模型定义（与 EM 训练一致）
# ==========================================
class AdaptiveInferenceModel(nn.Module):
    def __init__(self, num_classes, use_v1=True):
        super().__init__()
        if use_v1:
            swin = models.swin_t(weights=None)
        else:
            swin = models.swin_v2_t(weights=None)
        self.features = swin.features
        self.norm = swin.norm
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)          # Swin 输出 BHWC -> BCHW
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.head(x)


# ==========================================
# 2. V10 束搜索解码器（从 beam_decoder_v10.py 移植）
# ==========================================
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

        # 加载资产
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

        # PPMI 标准化参数
        self.ppmi_stats = {}
        for off in [1, 2, 3]:
            self.ppmi_stats[off] = {
                "mean": self.metadata[f"ppmi_{off}_mean"],
                "std": self.metadata[f"ppmi_{off}_std"]
            }
            if self.ppmi_stats[off]["std"] < 1e-8:
                self.ppmi_stats[off]["std"] = 1.0

        # MKN 参数
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

        # 折扣
        self.D1_b, self.D2_b, self.D3_b = self._compute_mkn_discounts(self.kn_stats["freq_of_freq_bigram"])
        self.D1_t, self.D2_t, self.D3_t = self._compute_mkn_discounts(self.kn_stats["freq_of_freq_trigram"])

        # Unigram
        self.unigram_probs = {ch: cnt / self.total_tokens for ch, cnt in self.unigram_counts.items()}

    def _compute_mkn_discounts(self, freq_of_freq):
        n1 = freq_of_freq.get(1, 0)
        n2 = freq_of_freq.get(2, 0)
        n3 = freq_of_freq.get(3, 0)
        n4 = freq_of_freq.get(4, 0)
        if n1 == 0:
            return 1.0, 2.0, 3.0
        Y = n1 / (n1 + 2 * n2) if (n1 + 2*n2) > 0 else 0.0
        D1 = 1 - 2 * Y * (n2 / n1) if n1 > 0 else 1.0
        D2 = 2 - 3 * Y * (n3 / n2) if n2 > 0 else 2.0
        D3 = 3 - 4 * Y * (n4 / n3) if n3 > 0 else 3.0
        return max(0.1, D1), max(0.1, D2), max(0.1, D3)

    def _get_discount(self, count, is_bigram=True):
        if is_bigram:
            if count == 1: return self.D1_b
            if count == 2: return self.D2_b
            return self.D3_b
        else:
            if count == 1: return self.D1_t
            if count == 2: return self.D2_t
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
        # 正常聚合
        max_k = min(3, len(seq))
        total = 0.0
        for k, lam in zip(range(1, max_k+1), self.ppmi_lambda):
            if k <= len(seq):
                score = self._get_ppmi_logprob(seq[-k], cand_char, offset=k)
                total += lam * score
        # offset=3 平滑回退
        if len(seq) >= 3:
            p3 = self._get_ppmi_logprob(seq[-3], cand_char, offset=3)
            p2 = self._get_ppmi_logprob(seq[-2], cand_char, offset=2)
            if p3 < -10:   # 原始 p3 极低
                lam3 = self.ppmi_lambda[2] if len(self.ppmi_lambda) > 2 else 0.2
                # 重算
                total = 0.0
                p1 = self._get_ppmi_logprob(seq[-1], cand_char, offset=1) if len(seq) >= 1 else np.log(self.epsilon)
                p2 = self._get_ppmi_logprob(seq[-2], cand_char, offset=2) if len(seq) >= 2 else np.log(self.epsilon)
                p3_smooth = 0.6 * p2 + 0.4 * p3
                total = self.ppmi_lambda[0]*p1 + self.ppmi_lambda[1]*p2 + self.ppmi_lambda[2]*p3_smooth
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
            D1, D2, D3 = self.D1_b, self.D2_b, self.D3_b
            lambda_ = (D1 * n1 + D2 * n2 + D3 * n3p) / context
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
            D1, D2, D3 = self.D1_t, self.D2_t, self.D3_t
            lambda_ = (D1 * n1 + D2 * n2 + D3 * n3p) / context
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

    def decode(self, all_candidates, beam_size=10):
        if not all_candidates:
            return ""

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

                    step_score = (self.alpha * visual_score) + \
                                 (self.beta * ppmi_sum) + \
                                 (self.gamma * trigram_log) + \
                                 (self.theta * bigram_log) + \
                                 (self.delta * mixture_score)

                    new_beams.append({
                        "seq": seq + [cand_char],
                        "score": old_score + step_score
                    })

            beams = sorted(new_beams, key=lambda x: x["score"], reverse=True)[:beam_size]

        best_beam = max(beams, key=lambda x: x["score"])
        return "".join(best_beam["seq"])


# ==========================================
# 3. 端到端推理引擎
# ==========================================
class AntiqueOCRSystemV10:
    def __init__(self, obb_model_path, classifier_model_path, vocab_path, assets_dir,
                 device='cuda', topk=5, beam_size=10):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.topk = topk
        self.beam_size = beam_size

        # 加载词表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        if isinstance(vocab, list):
            self.char2id = {char: idx for idx, char in enumerate(vocab)}
        else:
            self.char2id = vocab
        self.id2char = {idx: char for char, idx in self.char2id.items()}
        self.num_classes = len(self.char2id)

        # 初始化分类模型
        checkpoint = torch.load(classifier_model_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        clean_state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
        use_v1 = "features.1.0.attn.relative_position_bias_table" in clean_state_dict
        self.model = AdaptiveInferenceModel(num_classes=self.num_classes, use_v1=use_v1)
        self.model.load_state_dict(clean_state_dict, strict=False)
        self.model.to(self.device).eval()

        # 初始化解码器（加载 PPMI / KN 资产）
        self.decoder = UltimateBeamSearchDecoderV10(
            assets_dir=assets_dir,
            alpha=1.0, beta=0.25, gamma=0.45, theta=0.25, delta=0.30,
            eta=0.0, ppmi_scale=5.0, ppmi_lambda=(0.5, 0.3, 0.2)
        )

        # 检测器
        self.detector = YOLO(obb_model_path)

        # 图像预处理均值/标准差
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _sort_boxes(self, obb_data):
        """从右到左、从上到下排序（古文字阅读顺序）"""
        boxes = []
        for corners in obb_data:
            x_min, x_max = corners[:, 0].min(), corners[:, 0].max()
            y_min, y_max = corners[:, 1].min(), corners[:, 1].max()
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
            boxes.append({'cx': cx, 'cy': cy, 'rect': (x_min, y_min, x_max, y_max)})

        # 按列分组（x 坐标相近的为一列）
        cols = defaultdict(list)
        for b in boxes:
            col_id = int(b['cx'] // 60)   # 列宽阈值可调
            cols[col_id].append(b)

        # 从右到左排序列
        sorted_cols = sorted(cols.items(), key=lambda x: -x[0])
        sorted_boxes = []
        for _, col_items in sorted_cols:
            col_items.sort(key=lambda x: x['cy'])   # 从上到下
            sorted_boxes.extend(col_items)
        return sorted_boxes

    def _predict_candidates(self, crop_img):
        """对单张裁剪图像进行推理，返回 Top-K 候选 (char, prob)"""
        if crop_img is None or crop_img.size == 0:
            return [("<PAD_OR_UNK>", 1.0)]
        # resize 到 128x128（与训练一致）
        crop = cv2.resize(crop_img, (128, 128))
        # 归一化
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - self.mean) / self.std
        crop = np.transpose(crop, (2, 0, 1))
        tensor = torch.tensor(crop, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # 取 Top-K
        topk_idx = np.argsort(probs)[-self.topk:][::-1]
        candidates = [(self.id2char.get(i, "<PAD_OR_UNK>"), probs[i]) for i in topk_idx]
        return candidates

    def run(self, image_path):
        """对一张图片进行完整推理，返回 [{bbox, text}]"""
        full_img = cv2.imread(image_path)
        if full_img is None:
            return []

        # 检测 OBB
        results = self.detector.predict(image_path, conf=0.5, iou=0.3, verbose=False)[0]
        if results.obb is None or len(results.obb) == 0:
            return []

        obb_data = results.obb.xyxyxyxy.cpu().numpy()
        sorted_boxes = self._sort_boxes(obb_data)

        # 裁剪并获取候选
        all_candidates = []
        bboxes = []
        for b in sorted_boxes:
            x_min, y_min, x_max, y_max = b['rect']
            # 加 padding
            pad = 5
            x1 = max(0, int(x_min - pad))
            y1 = max(0, int(y_min - pad))
            x2 = min(full_img.shape[1], int(x_max + pad))
            y2 = min(full_img.shape[0], int(y_max + pad))
            crop = full_img[y1:y2, x1:x2]
            candidates = self._predict_candidates(crop)
            all_candidates.append(candidates)
            bboxes.append([int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)])

        # 束搜索解码
        if not all_candidates:
            return []
        decoded_seq = self.decoder.decode(all_candidates, beam_size=self.beam_size)
        # 如果解码长度不足，用 <PAD_OR_UNK> 补齐
        if len(decoded_seq) < len(bboxes):
            decoded_seq += "<PAD_OR_UNK>" * (len(bboxes) - len(decoded_seq))
        elif len(decoded_seq) > len(bboxes):
            decoded_seq = decoded_seq[:len(bboxes)]

        result = [{"bbox": bbox, "text": char} for bbox, char in zip(bboxes, decoded_seq)]
        return result


# ==========================================
# 4. 主程序
# ==========================================
def main():
    # ---- 配置（请根据实际路径修改） ----
    CONFIG = {
        "OBB_MODEL": "/app/models/last.pt",
        "CLASSIFIER_MODEL": "/app/models/em_iter_1.pt",      # 最佳 EM 迭代权重
        "VOCAB_PATH": "/app/models/vocab.json",
        "ASSETS_DIR": "/saisdata/50/eval/images/",              # 包含 ppmi_*.pkl, kn_stats.pkl 等
        "INPUT_DIR": "/saisdata/50/eval/images/",
        "OUTPUT_JSON": "/saisresult/prediction.json",
        "DEVICE": "cuda",
        "TOPK": 5,
        "BEAM_SIZE": 10
    }

    print("🔄 初始化 OCR 系统 V10...")
    ocr = AntiqueOCRSystemV10(
        obb_model_path=CONFIG["OBB_MODEL"],
        classifier_model_path=CONFIG["CLASSIFIER_MODEL"],
        vocab_path=CONFIG["VOCAB_PATH"],
        assets_dir=CONFIG["ASSETS_DIR"],
        device=CONFIG["DEVICE"],
        topk=CONFIG["TOPK"],
        beam_size=CONFIG["BEAM_SIZE"]
    )

    # 获取所有 PNG
    #image_files = [f for f in os.listdir(CONFIG["INPUT_DIR"]) if f.lower().endswith('.png')]
    image_files = [f for f in os.listdir(CONFIG["INPUT_DIR"])]
    image_files.sort()
    print(f"🚀 共 {len(image_files)} 张图片待处理")

    final_predictions = {}
    for idx, img_name in enumerate(image_files):
        img_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(CONFIG["INPUT_DIR"], img_name)
        try:
            result = ocr.run(img_path)
            final_predictions[img_id] = result
            if (idx + 1) % 10 == 0:
                print(f"进度: {idx + 1}/{len(image_files)}")
        except Exception as e:
            print(f"❌ 处理 {img_name} 出错: {e}")
            final_predictions[img_id] = []

    os.makedirs(os.path.dirname(CONFIG["OUTPUT_JSON"]), exist_ok=True)
    with open(CONFIG["OUTPUT_JSON"], 'w', encoding='utf-8') as f:
        json.dump(final_predictions, f, ensure_ascii=False, indent=2)
    print(f"✅ 提交文件已生成: {CONFIG['OUTPUT_JSON']}")


if __name__ == "__main__":
    main()
