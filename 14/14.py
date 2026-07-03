import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ==================== 配置参数 ====================
PATTERN_SIZE = (9, 6)           # 内角点 (列, 行)
SQUARE_SIZE = 25.0              # 方格边长 (mm)
IMAGE_FOLDER = "raw"            # 图片文件夹（相对于脚本所在目录）
UNDISTORT_INDEX = None          # 去畸变示例图片索引（自动取中间一张）

# ==================== 辅助函数：支持中文路径读取 ====================
def imread_unicode(file_path):
    """使用 imdecode 读取含中文路径的图片"""
    try:
        with open(file_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"读取失败: {file_path}, 错误: {e}")
        return None

# ==================== 步骤1：读取所有标定图片 ====================
# 使用 pathlib 遍历，自动处理编码
image_folder = Path(IMAGE_FOLDER)
if not image_folder.exists():
    raise FileNotFoundError(f"文件夹 '{IMAGE_FOLDER}' 不存在，请检查。")

# 收集所有常见图片格式
extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
image_paths = sorted([p for p in image_folder.glob("*") if p.suffix.lower() in extensions])

if not image_paths:
    raise FileNotFoundError(f"在 '{IMAGE_FOLDER}' 中未找到图片文件。")

print(f"发现 {len(image_paths)} 张图片，开始处理...")

objpoints = []   # 世界坐标 (3D)
imgpoints = []   # 图像坐标 (2D)

# 生成棋盘格角点的世界坐标 (Z=0)
objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

img_size = None

# 用于去畸变的索引（取中间一张成功检测的）
successful_indices = []

for i, img_path in enumerate(image_paths):
    img = imread_unicode(img_path)
    if img is None:
        print(f"警告：无法读取 {img_path.name}，跳过")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_size is None:
        img_size = gray.shape[::-1]   # (width, height)

    # 检测棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)
    if ret:
        # 亚像素精度优化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners_sub)

        # 绘制角点并保存（文件名不含中文）
        cv2.drawChessboardCorners(img, PATTERN_SIZE, corners_sub, ret)
        out_name = f"detected_corners_{i:03d}.jpg"
        cv2.imwrite(out_name, img)
        successful_indices.append(i)
        print(f"图片 {i+1}: {img_path.name} 检测成功")
    else:
        print(f"图片 {i+1}: {img_path.name} 检测失败，跳过")

print(f"共检测到 {len(objpoints)} 张有效图片")

if len(objpoints) < 3:
    raise ValueError("有效图片少于3张，无法标定。请确认：\n"
                     "1) 图片确实包含棋盘格（内角点 9×6）\n"
                     "2) 图片清晰、无过度反光或模糊\n"
                     "3) 棋盘格完整可见")

# 确定去畸变示例图片索引（从成功检测的图片中取中间一张）
if successful_indices:
    UNDISTORT_INDEX = successful_indices[len(successful_indices)//2]
else:
    UNDISTORT_INDEX = None

# ==================== 步骤3：相机标定 ====================
ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None
)

# ==================== 步骤4：计算重投影误差 ====================
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

mean_error = total_error / len(objpoints)
print("\n========== 标定结果 ==========")
print(f"相机内参矩阵 K:\n{K}")
print(f"畸变系数 D (k1, k2, p1, p2, k3):\n{dist_coeffs.ravel()}")
print(f"重投影误差 (RMS): {mean_error:.4f} 像素")

# ==================== 步骤5：去畸变示例 ====================
if UNDISTORT_INDEX is not None and UNDISTORT_INDEX < len(image_paths):
    sample_path = image_paths[UNDISTORT_INDEX]
    sample_img = imread_unicode(sample_path)
    if sample_img is not None:
        h, w = sample_img.shape[:2]
        dst = cv2.undistort(sample_img, K, dist_coeffs, None, K)
        cv2.imwrite("original_sample.jpg", sample_img)
        cv2.imwrite("undistorted_sample.jpg", dst)
        print(f"去畸变示例已保存：original_sample.jpg 和 undistorted_sample.jpg")

        # 显示对比（若支持图形界面）
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
            ax1.set_title("原始图像")
            ax2.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
            ax2.set_title("去畸变后")
            plt.show()
        except Exception as e:
            print("无法显示图像（可能无图形界面），请查看保存的文件。")
else:
    print("未找到用于去畸变的示例图片。")

# ==================== 步骤6：保存结果到文本文件 ====================
with open("calibration_results.txt", "w", encoding="utf-8") as f:
    f.write("========== 相机标定结果 ==========\n")
    f.write(f"棋盘格内角点：{PATTERN_SIZE[0]} × {PATTERN_SIZE[1]}\n")
    f.write(f"方格边长：{SQUARE_SIZE} mm\n")
    f.write(f"图像分辨率：{img_size[0]} × {img_size[1]}\n")
    f.write(f"有效标定图片数：{len(objpoints)}\n\n")
    f.write("内参矩阵 K:\n")
    np.savetxt(f, K, fmt="%.4f")
    f.write("\n畸变系数 D (k1, k2, p1, p2, k3):\n")
    np.savetxt(f, dist_coeffs.reshape(1, -1), fmt="%.6f")
    f.write(f"\n重投影误差 (RMS)：{mean_error:.4f} 像素\n")
    f.write("\n标定完成时间：自动生成\n")

print("\n所有结果已保存至 calibration_results.txt")
print("检测到的角点图像已保存为 detected_corners_*.jpg")