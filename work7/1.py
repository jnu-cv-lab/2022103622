import cv2
import numpy as np

# ================== 读取图像 ==================
img_box = cv2.imread("box.png")
img_scene = cv2.imread("box_in_scene.png")

# ================== 任务一：ORB 特征检测 ==================
orb = cv2.ORB_create(nfeatures=1000)
kp_box, des_box = orb.detectAndCompute(img_box, None)
kp_scene, des_scene = orb.detectAndCompute(img_scene, None)

img_box_kp = cv2.drawKeypoints(img_box, kp_box, None)
img_scene_kp = cv2.drawKeypoints(img_scene, kp_scene, None)

cv2.imwrite("box_keypoints.png", img_box_kp)
cv2.imwrite("scene_keypoints.png", img_scene_kp)

print("===== 任务一 结果 =====")
print(f"box 关键点数量：{len(kp_box)}")
print(f"scene 关键点数量：{len(kp_scene)}")
print(f"描述子维度：{des_box.shape[1]}")

# ================== 任务二：ORB 特征匹配 ==================
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_box, des_scene)
matches = sorted(matches, key=lambda x: x.distance)

img_match = cv2.drawMatches(img_box, kp_box, img_scene, kp_scene, matches[:30], None)
cv2.imwrite("orb_matches.png", img_match)

print("\n===== 任务二 结果 =====")
print(f"总匹配对数：{len(matches)}")

# ================== 任务三：RANSAC 剔除误匹配 ==================
src_pts = np.float32([kp_box[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matches_mask = mask.ravel().tolist()
inlier_num = sum(matches_mask)

img_ransac = cv2.drawMatches(img_box, kp_box, img_scene, kp_scene,
                            [m for i, m in enumerate(matches) if matches_mask[i]], None)
cv2.imwrite("ransac_matches.png", img_ransac)

print("\n===== 任务三 结果 =====")
print(f"Homography 矩阵：\n{H}")
print(f"RANSAC 内点数量：{inlier_num}")
print(f"内点比例：{inlier_num / len(matches):.4f}")

# ================== 任务四：目标定位与框选（续写完成） ==================
# 1. 获取原图 box 的四个角点
h, w = img_box.shape[:2]
box_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

# 2. 使用单应矩阵 H 映射到 scene 图中
scene_corners = cv2.perspectiveTransform(box_corners, H)

# 3. 在 scene 图上绘制包围框（闭合四边形）
img_result = img_scene.copy()
img_result = cv2.polylines(img_result, [np.int32(scene_corners)], True, (0, 255, 0), 3)

# 4. 保存最终定位结果
cv2.imwrite("target_detection_result.png", img_result)

print("\n===== 任务四 结果 =====")
print("目标定位完成！")
print("已在 scene 图中用绿色矩形框出目标物体")
print("结果已保存为：target_detection_result.png")

# ================== 显示所有窗口 ==================
cv2.imshow("Task1: Box Keypoints", img_box_kp)
cv2.imshow("Task1: Scene Keypoints", img_scene_kp)
cv2.imshow("Task2: ORB Matches", img_match)
cv2.imshow("Task3: RANSAC Matches", img_ransac)
cv2.imshow("Task4: Target Detection", img_result)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ================== 任务六：参数对比实验 ==================
def run_orb_test(nfeatures):
    # 重建ORB，修改参数
    orb_test = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb_test.detectAndCompute(img_box, None)
    kp2, des2 = orb_test.detectAndCompute(img_scene, None)

    # 特征匹配
    bf_test = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_test = bf_test.match(des1, des2)
    matches_test = sorted(matches_test, key=lambda x: x.distance)

    # RANSAC单应矩阵
    src = np.float32([kp1[m.queryIdx].pt for m in matches_test]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches_test]).reshape(-1,1,2)
    H_test, mask_test = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    inlier_count = int(np.sum(mask_test))
    total_match = len(matches_test)
    inlier_ratio = inlier_count / total_match if total_match > 0 else 0

    # 目标定位是否成功
    locate_ok = True
    if H_test is None:
        locate_ok = False

    return len(kp1), len(kp2), total_match, inlier_count, round(inlier_ratio,4), locate_ok

# 测试三组参数
param_list = [500, 1000, 2000]
result_table = []

print("\n========== 任务六：参数对比实验结果 ==========")
print(f"{'nfeatures':<10}{'模板关键点':<12}{'场景关键点':<12}{'匹配数量':<10}{'内点数量':<12}{'内点比例':<10}{'是否成功定位'}")
for n in param_list:
    kp_num1, kp_num2, match_num, inlier_num, ratio, ok_flag = run_orb_test(n)
    result_table.append([n,kp_num1,kp_num2,match_num,inlier_num,ratio,ok_flag])
    ok_text = "是" if ok_flag else "否"
    print(f"{n:<10}{kp_num1:<12}{kp_num2:<12}{match_num:<10}{inlier_num:<12}{ratio:<10}{ok_text}")