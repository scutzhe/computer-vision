//stretchlim函数头文件

#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <cassert>

using namespace cv;
using namespace std;

vector<int> calcGrayLevel(Mat &img);                       //计算灰度级，即算出从0到255区间的任意一个灰度值i，在图像img中有多少个像素点的灰度值为i
vector<double> pdf(vector<int> gray_level, Mat &img); //计算概率密度pdf
vector<double> cdf(vector<double> pdf);                   //计算概率分布cdf
double findLow(double input_low, vector<double> cdf);
double findHihg(double input_high, vector<double> cdf);
vector<double> strecthlim(Mat &img, double tol_low = 0.01, double tol_high = 0.99);

vector<int> calcGrayLevel_32FC(Mat &img);

void test_function_strecthlim();