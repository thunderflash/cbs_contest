import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import json
import pickle
from collections import defaultdict
from scipy.spatial import KDTree
from torchvision.models import swin_t
from ultralytics import YOLO

# ==========================================
# 1. PPMI 重排序模块
# ==========================================
class PPMIReranker:
    def __init__(self, ppmi_matrix, alpha=0.8, beta=0.2):
        self.ppmi = ppmi_matrix
        self.alpha = alpha
        self.beta = beta

    def score_candidate(self, candidate_char, neighbors):
        context_score = 0
        for c, p in neighbors:
            if c is None: continue
            ppmi_val = self.ppmi.get(candidate_char, {}).get(c, 0.0)
            context_score += p * ppmi_val
        return np.tanh(context_score)

    def rerank(self, top5_list, neighbors):
        best_char = None
        best_score = -1e9
        for char, prob in top5_list:
            context = self.score_candidate(char, neighbors)
            final_score = (self.alpha * prob + self.beta * context)
            if final_score > best_score:
                best_score = final_score
                best_char = char
        return best_char

# ==========================================
# 2. 分类与空间邻居纠错引擎
# ==========================================
class AntiqueCharRefiner:
    def __init__(self, model_path, vocab_path, corpus_path, ppmi_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
            unique_chars = sorted(list(set(manifest.values())))
            self.char2id = {char: i + 1 for i, char in enumerate(unique_chars)}
            self.char2id["<PAD_OR_UNK>"] = 0
            self.id2char = {v: k for k, v in self.char2id.items()}
        
        self.num_classes = len(self.char2id) 
        self.model = swin_t(weights=None)
        self.model.head = nn.Linear(self.model.head.in_features, self.num_classes)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        
        with open(ppmi_path, 'rb') as f:
            ppmi = pickle.load(f)
            self.ppmi_reranker = PPMIReranker(ppmi_matrix=ppmi)

    def predict_with_spatial_context(self, img_paths, bboxes):
        if not img_paths: return []
        
        tree = KDTree(np.array(bboxes))
        all_candidates = []
        
        for p in img_paths:
            img = cv2.imread(p)
            if img is None:
                all_candidates.append([("<PAD_OR_UNK>", 1.0)])
                continue
            img = cv2.resize(img, (128, 128))
            img_tensor = torch.tensor(
                np.transpose(img.astype(np.float32)/255.0, (2, 0, 1))
            ).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                probs = torch.softmax(self.model(img_tensor), dim=1)[0].cpu().numpy()
            
            top_k = np.argsort(probs)[-5:][::-1]
            all_candidates.append([(self.id2char.get(i, "<PAD_OR_UNK>"), probs[i]) for i in top_k])
        
        final_results = []
        for i, candidates in enumerate(all_candidates):
            k_val = min(len(bboxes), 6)
            _, indices = tree.query(bboxes[i], k=k_val)
            if np.isscalar(indices): indices = [indices]
            
            neighbors = []
            for idx in indices:
                if idx >= len(all_candidates): continue
                for c, p in all_candidates[idx][:2]:
                    neighbors.append((c, p))
            
            best_char = self.ppmi_reranker.rerank(candidates, neighbors)
            final_results.append(best_char)

        return final_results

# ==========================================
# 3. 端到端系统
# ==========================================
class AntiqueOCRSystem:
    def __init__(self, obb_model_path, classifier):
        self.detector = YOLO(obb_model_path)
        self.classifier = classifier

    def _sort_boxes(self, obb_data):
        boxes_with_centers = []
        for corners in obb_data:
            x_min, x_max = corners[:, 0].min(), corners[:, 0].max()
            y_min, y_max = corners[:, 1].min(), corners[:, 1].max()
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
            boxes_with_centers.append({'cx': cx, 'cy': cy, 'rect': (x_min, y_min, x_max, y_max)})
        
        cols = defaultdict(list)
        for b in boxes_with_centers:
            col_id = int(b['cx'] // 60) 
            cols[col_id].append(b)
        
        sorted_col_ids = sorted(cols.keys(), reverse=True)
        sorted_boxes = []
        for cid in sorted_col_ids:
            col_items = sorted(cols[cid], key=lambda x: x['cy'])
            sorted_boxes.extend(col_items)
        return sorted_boxes

    def run(self, image_path):
        full_img = cv2.imread(image_path)
        if full_img is None: return []
        
        if len(full_img.shape) == 3 and full_img.shape[2] == 4:
            full_img = cv2.cvtColor(full_img, cv2.COLOR_BGRA2BGR)

        results = self.detector.predict(image_path, conf=0.5, iou=0.3, verbose=False)[0]
        if results.obb is None or len(results.obb) == 0:
            return []

        obb_data = results.obb.xyxyxyxy.cpu().numpy()
        sorted_boxes = self._sort_boxes(obb_data)
        
        crops = []
        centers = []
        # 存储原始 BBox 以便最后输出 [x, y, w, h]
        final_bboxes = [] 

        for i, b in enumerate(sorted_boxes):
            x_min, y_min, x_max, y_max = b['rect']
            
            # 1. 计算提交要求的 [x, y, w, h] (整数)
            w = int(x_max - x_min)
            h = int(y_max - y_min)
            final_bboxes.append([int(x_min), int(y_min), w, h])

            # 2. 裁剪切片用于识别 (含 Padding)
            pad = 5
            cx1, cy1 = max(0, int(x_min-pad)), max(0, int(y_min-pad))
            cx2, cy2 = min(full_img.shape[1], int(x_max+pad)), min(full_img.shape[0], int(y_max+pad))
            crop = full_img[cy1:cy2, cx1:cx2]
            
            temp_path = f"tmp_crop_{os.getpid()}_{i}.jpg"
            cv2.imwrite(temp_path, crop)
            crops.append(temp_path)
            centers.append((b['cx'], b['cy']))
        
        recognized_chars = self.classifier.predict_with_spatial_context(crops, centers)
        
        for p in crops: 
            if os.path.exists(p): os.remove(p)
            
        # 返回 [{ "bbox": [...], "text": "..." }, ...]
        return [{"bbox": bbox, "text": char} for bbox, char in zip(final_bboxes, recognized_chars)]

# ==========================================
# 4. 执行主程序
# ==========================================
def main():
    # --- 提交规范配置 ---
    # run_inference.py 中的 CONFIG 示例
    CONFIG = {
        "OBB_MODEL": "/app/models/last.pt",              # 对应 COPY models/ /app/models/
        "SWIN_MODEL": "/app/models/pure_swin_classifier.pt",
        "VOCAB_JSON": "/app/models/unified_clean_manifest.json",
        "PPMI_PATH": "/app/models/ppmi.pkl",
        "CORPUS_TXT": "/app/models/jinwen_corpus.txt",
        "INPUT_DIR": "/saisdata/50/eval/images/",        # 赛题指定输入路径
        "OUTPUT_JSON": "/saisresult/prediction.json",    # 赛题指定输出路径
        "DEVICE": "cuda"
    }

    print("🔄 初始化系统...")
    refiner = AntiqueCharRefiner(
        model_path=CONFIG["SWIN_MODEL"],
        vocab_path=CONFIG["VOCAB_JSON"],
        corpus_path=CONFIG["CORPUS_TXT"],
        ppmi_path=CONFIG["PPMI_PATH"],
        device=CONFIG["DEVICE"]
    )
    
    ocr_system = AntiqueOCRSystem(
        obb_model_path=CONFIG["OBB_MODEL"],
        classifier=refiner
    )

    # 获取所有 PNG 图片
    image_files = [f for f in os.listdir(CONFIG["INPUT_DIR"]) if f.lower().endswith('.png')]
    image_files.sort()

    final_predictions = {}
    print(f"🚀 开始处理 {len(image_files)} 张图片...")

    for idx, img_name in enumerate(image_files):
        img_id = os.path.splitext(img_name)[0] # 获取图片ID
        img_path = os.path.join(CONFIG["INPUT_DIR"], img_name)
        
        try:
            res = ocr_system.run(img_path)
            final_predictions[img_id] = res
            if (idx + 1) % 10 == 0:
                print(f"进度: {idx + 1}/{len(image_files)}")
        except Exception as e:
            print(f"❌ 处理 {img_name} 出错: {e}")
            final_predictions[img_id] = []

    # 确保输出目录存在
    os.makedirs(os.path.dirname(CONFIG["OUTPUT_JSON"]), exist_ok=True)

    # 保存为 JSON (UTF-8)
    with open(CONFIG["OUTPUT_JSON"], 'w', encoding='utf-8') as f:
        json.dump(final_predictions, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 提交文件已生成: {CONFIG['OUTPUT_JSON']}")

if __name__ == "__main__":
    main()