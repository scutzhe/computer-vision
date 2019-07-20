#pragma once

#include <opencv2/opencv.hpp>
using namespace cv;

Mat MSRCR(const Mat& I);
Mat MSR(const Mat& I, int lowScale = 25, int medScale = 100, int highScale = 240);
Mat SSR(const Mat &I, int arg);
Mat multiscaleRetinex(const Mat& matin);

//计算每个像素值的log值
void MatLog(Mat& mat);
//计算Mat与一个数相加
Mat MatAddNum(const Mat& m, double num);
//计算Mat与一个数相乘
void MatMultiNum(Mat& m, double num);
//对应像素相乘
void MatMul(Mat& a, const Mat& b);
//打印出来Mat的部分像素值，用作测试
void PresentFirstrow(const Mat& m);
//获取图片中单通道数据
Mat GetSingleChannel(const Mat& img, int ch);