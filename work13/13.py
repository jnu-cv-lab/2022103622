import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mediapipe as mp
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# -------------------- 配置参数 --------------------
DATA_ROOT = "./archive"                # 数据集根目录（包含6个子文件夹）
TARGET_FRAMES = 30                     # 每段视频统一帧数
KEYPOINT_DIM = 132                     # 33个关键点 × 4特征 (x,y,z,visibility)
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
NUM_CLASSES = 6
DROPOUT = 0.1
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 类别映射
LABEL_MAP = {
    "forehand_drive": 0,
    "forehand_lift": 1,
    "forehand_net_shot": 2,
    "forehand_clear": 3,
    "backhand_drive": 4,
    "backhand_net_shot": 5
}
# 反向映射用于推理输出
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# -------------------- MediaPipe 关键点提取 --------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def extract_pose_sequence(video_path):
    """
    从视频中提取每一帧的人体关键点，返回形状 (frames, 132) 的数组
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")
    
    frames_data = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # MediaPipe 需要 RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            # 提取33个关键点的 x,y,z,visibility
            landmarks = results.pose_landmarks.landmark
            frame_vec = []
            for lm in landmarks:
                frame_vec.extend([lm.x, lm.y, lm.z, lm.visibility])
            frames_data.append(frame_vec)
        else:
            # 若未检测到人体，则填充零向量（可根据需要调整）
            frames_data.append([0.0] * KEYPOINT_DIM)
    cap.release()
    return np.array(frames_data, dtype=np.float32)  # (F, 132)

# -------------------- 序列重采样与归一化 --------------------
def normalize_pose(seq):
    """
    归一化：以左右髋部中心为原点，以肩宽进行尺度归一化
    seq: (F, 132)
    返回归一化后的序列
    """
    # 关键点索引（MediaPipe Pose）
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    
    # 提取坐标 (x,y) 从每帧，形状 (F, 33, 2)
    F = seq.shape[0]
    coords = seq[:, :33*2].reshape(F, 33, 2)  # 取x,y，忽略z和visibility用于归一化
    
    # 髋部中心
    hip_center = (coords[:, LEFT_HIP, :] + coords[:, RIGHT_HIP, :]) / 2.0  # (F,2)
    # 肩宽
    shoulder_width = np.linalg.norm(coords[:, LEFT_SHOULDER, :] - coords[:, RIGHT_SHOULDER, :], axis=1, keepdims=True)  # (F,1)
    shoulder_width = np.where(shoulder_width > 1e-6, shoulder_width, 1.0)  # 避免除零
    
    # 对每一帧进行平移和缩放
    normalized = np.zeros_like(seq)
    for f in range(F):
        # 平移：所有关键点减去髋部中心
        centered = coords[f] - hip_center[f]  # (33,2)
        # 缩放：除以肩宽
        scaled = centered / shoulder_width[f]
        # 写回x,y, z和visibility保持不变（但z也需要类似处理？此处只对x,y归一化，z可保留相对尺度，但也可以做类似处理，保持一致性）
        # 简单起见，只归一化x,y，z和visibility保持原样
        normalized[f, :33*2] = scaled.reshape(-1)
        # 保留z和visibility
        normalized[f, 33*2:] = seq[f, 33*2:]
    return normalized

def resample_sequence(seq, target_len):
    """
    将序列重采样为固定长度，使用均匀采样（线性插值可选，这里用最近邻索引）
    seq: (F, D)
    """
    F = seq.shape[0]
    if F == 0:
        return np.zeros((target_len, seq.shape[1]), dtype=np.float32)
    if F == target_len:
        return seq
    indices = np.linspace(0, F-1, target_len, dtype=int)
    return seq[indices]

# -------------------- 数据预处理--------------------
def preprocess_dataset(data_root, save_dir="./processed"):
    """
    遍历数据集，提取关键点，重采样，归一化，保存为npy
    """
    os.makedirs(save_dir, exist_ok=True)
    all_data = []
    all_labels = []
    
    # 获取所有类别文件夹
    for class_name in LABEL_MAP.keys():
        class_dir = os.path.join(data_root, class_name)
        if not os.path.isdir(class_dir):
            print(f"警告：目录不存在 {class_dir}")
            continue
        label = LABEL_MAP[class_name]
        video_files = [f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        print(f"处理类别 {class_name}，共 {len(video_files)} 个视频")
        for vid_file in tqdm(video_files, desc=class_name):
            vid_path = os.path.join(class_dir, vid_file)
            try:
                seq_raw = extract_pose_sequence(vid_path)
                # 重采样
                seq_resampled = resample_sequence(seq_raw, TARGET_FRAMES)
                # 归一化
                seq_normalized = normalize_pose(seq_resampled)
                all_data.append(seq_normalized)
                all_labels.append(label)
            except Exception as e:
                print(f"处理 {vid_path} 时出错: {e}")
                continue
    
    if len(all_data) == 0:
        raise RuntimeError("没有成功提取任何数据，请检查数据集路径和视频格式。")
    
    X = np.array(all_data, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    # 划分训练集和测试集 (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 保存
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)
    
    # 保存标签映射
    with open(os.path.join(save_dir, "label_map.json"), "w") as f:
        json.dump(LABEL_MAP, f, indent=2)
    
    print(f"预处理完成！训练集 {X_train.shape[0]} 样本，测试集 {X_test.shape[0]} 样本")
    print(f"数据已保存至 {save_dir}")
    return X_train, X_test, y_train, y_test

# -------------------- 自定义 Dataset --------------------
class SkeletonDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------- Transformer 模型 --------------------
class SkeletonTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, TARGET_FRAMES, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x):
        # x: (B, T, input_dim)
        x = self.embedding(x)  # (B, T, d_model)
        x = x + self.pos_embedding
        x = self.transformer(x)  # (B, T, d_model)
        # 全局平均池化
        x = x.mean(dim=1)        # (B, d_model)
        logits = self.classifier(x)
        return logits

# -------------------- 训练函数 --------------------
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels

# -------------------- 完整训练流程 --------------------
def train_model(X_train, y_train, X_test, y_test):
    train_dataset = SkeletonDataset(X_train, y_train)
    test_dataset = SkeletonDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SkeletonTransformer(
        input_dim=KEYPOINT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("开始训练...")
    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion)
        print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    # 最终测试评估
    _, _, preds, true_labels = evaluate(model, test_loader, criterion)
    print("\n=== 最终测试结果 ===")
    print(f"准确率: {accuracy_score(true_labels, preds):.4f}")
    print("混淆矩阵:")
    print(confusion_matrix(true_labels, preds))
    print("分类报告:")
    print(classification_report(true_labels, preds, target_names=list(LABEL_MAP.keys())))
    
    # 保存模型
    torch.save(model.state_dict(), "skeleton_transformer.pth")
    print("模型已保存至 skeleton_transformer.pth")
    return model

# -------------------- 推理函数（单个视频） --------------------
def predict_video(video_path, model_path="skeleton_transformer.pth"):
    """
    对单个视频进行动作分类
    """
    # 加载模型
    model = SkeletonTransformer(
        input_dim=KEYPOINT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 提取骨架
    seq_raw = extract_pose_sequence(video_path)
    if seq_raw.shape[0] == 0:
        print("视频中未检测到人体关键点，无法预测。")
        return None
    seq_resampled = resample_sequence(seq_raw, TARGET_FRAMES)
    seq_normalized = normalize_pose(seq_resampled)
    # 添加batch维度
    input_tensor = torch.tensor(seq_normalized, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    
    pred_label = IDX_TO_LABEL[pred_class]
    print(f"Predicted class: {pred_label}")
    print(f"Confidence: {confidence:.4f}")
    return pred_label, confidence

# -------------------- 主程序 --------------------
if __name__ == "__main__":
    # 1. 预处理
    processed_dir = "./processed"
    if not os.path.exists(os.path.join(processed_dir, "X_train.npy")):
        print("开始预处理视频数据...")
        X_train, X_test, y_train, y_test = preprocess_dataset(DATA_ROOT, processed_dir)
    else:
        print("加载已处理的数据...")
        X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
        X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
        y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
        y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
        print(f"训练集 {X_train.shape[0]} 样本，测试集 {X_test.shape[0]} 样本")
    
    # 2. 训练
    model = train_model(X_train, y_train, X_test, y_test)
    
    # 3. 推理示例
    demo_video = "./demo_video.mp4"
    if os.path.exists(demo_video):
        print("\n=== 推理演示 ===")
        predict_video(demo_video)
    else:
        print("\n未找到演示视频，跳过推理演示。")

    print("所有任务完成！")