 #pragma once
//imadjust函数头文件

#include<opencv2/opencv.hpp>
#include<iostream>
#include<cassert>
#include<vector>

using namespace cv;

void imadjust(Mat& input, Mat& output, double low_in = 0.0, double high_in  = 1.0, double low_out = 0.0, double high_out = 1.0, double gamma = 1);//matlab,像素区间[0,1]
void imadjust(Mat& input, Mat& output, std::vector<double> in = { 0.0, 1.0 }, double low_out = 0.0, double high_out = 1.0, double gamma = 1);
void imadjust2(Mat& input, Mat& output, int low_in, int high_in, int low_out, int high_out, double gamma = 1);//opencv，像素区间[0,255]
std::vector<uchar> gammaLut(const double gamma, const double c = 1.0);//灰度值的伽马变换结果表lut
bool is0to1(const double var);

void test_function_imadjust();