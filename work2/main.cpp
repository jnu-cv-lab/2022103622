#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 1. 读取图片（确保1.jpg和程序在同一目录）
    Mat img = imread("/mnt/c/Users/enlong/Desktop/1.jpg");
    if (img.empty()) {
        cout << "错误：无法读取图片 1.jpg，请检查文件路径！" << endl;
        return -1;
    }

    // 2. 输出图像基本信息
    cout << "===== 图像基本信息 =====" << endl;
    cout << "图像尺寸（宽度 x 高度）: " << img.cols << " x " << img.rows << endl;
    cout << "图像通道数: " << img.channels() << " (3为彩色, 1为灰度)" << endl;
    cout << "图像数据类型: " << img.type() << " (通常16对应CV_8UC3)" << endl;

    // 3. 显示原图
    imshow("【任务3】原图显示", img);
    waitKey(0); // 按任意键继续

    // 4. 转换为灰度图并显示
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY); // BGR 转灰度
    imshow("【任务4】灰度图转换", gray);
    waitKey(0);

    // 5. 保存灰度图
    imwrite("gray_result.jpg", gray);
    cout << "\n灰度图已保存为: gray_result.jpg" << endl;

    // 6. 模拟NumPy操作：像素访问 + 区域裁剪
    cout << "\n===== 模拟NumPy操作 =====" << endl;
    // 输出左上角像素值
    Vec3b pixel = img.at<Vec3b>(0, 0);
    cout << "左上角像素 (B, G, R): " 
         << (int)pixel[0] << ", " 
         << (int)pixel[1] << ", " 
         << (int)pixel[2] << endl;

    // 裁剪左上角50x50区域
    Rect roi(0, 0, 50, 50);
    Mat cropped = img(roi);
    imshow("【任务6】左上角裁剪", cropped);
    imwrite("cropped_top_left.jpg", cropped);
    cout << "裁剪区域已保存为: cropped_top_left.jpg" << endl;

    waitKey(0);
    destroyAllWindows();
    return 0;
}