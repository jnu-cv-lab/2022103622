import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 任务1：读取图片1.jpg
img_path = "/mnt/c/Users/enlong/Desktop/1.jpg"
img = cv2.imread(img_path)

# 判断图片是否读取成功（排查路径/格式错误）
if img is None:
    print("❌ 图片读取失败！请检查：1.图片名是否为1.jpg 2.图片是否在桌面 3.WSL路径是否正确")
    exit()

# 任务2：输出图像基本信息（宽度、高度、通道数、数据类型）
height, width, channels = img.shape
dtype = img.dtype
print("="*30)
print(f"图像宽度(列)：{width} 像素")
print(f"图像高度(行)：{height} 像素")
print(f"图像通道数：{channels}（彩色图为3，灰度图为1）")
print(f"图像数据类型：{dtype}（通常为uint8）")
print("="*30)

# 任务3：OpenCV(BGR)转Matplotlib(RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 任务4：转换为灰度图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 任务5：保存灰度图（保存至桌面，方便查找）
gray_save_path = "/mnt/c/Users/enlong/Desktop/gray_1.jpg"
cv2.imwrite(gray_save_path, gray_img)
print(f"✅ 灰度图已保存至桌面：gray_1.jpg")

# 任务6：NumPy简单操作（像素值读取+左上角区域裁剪保存）
# 6.1 读取灰度图坐标(50,50)的像素值
pixel_val = gray_img[50, 50]
print(f"✅ 灰度图(50,50)像素值：{pixel_val}")

# 6.2 裁剪左上角200×200区域
crop_size = 200
if height >= crop_size and width >= crop_size:
    crop_img = img[0:crop_size, 0:crop_size]
    crop_save_path = "/mnt/c/Users/enlong/Desktop/crop_1.jpg"
    cv2.imwrite(crop_save_path, crop_img)
    print(f"✅ 左上角{crop_size}×{crop_size}裁剪图已保存至桌面：crop_1.jpg")
else:
    crop_img = img[0:height, 0:width]
    crop_save_path = "/mnt/c/Users/enlong/Desktop/crop_1.jpg"
    cv2.imwrite(crop_save_path, crop_img)
    print(f"✅ 图片尺寸不足{crop_size}×{crop_size}，已保存完整图至桌面：crop_1.jpg")

print("="*30)
print("✅ 所有实验任务执行完成！")
print("📁 桌面生成文件：gray_1.jpg（灰度图）、crop_1.jpg（裁剪图）")
print("="*30)