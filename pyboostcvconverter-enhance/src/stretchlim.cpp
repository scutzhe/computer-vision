//stretchlim函数实现

#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "stretchlim.h"
#include <iostream>
using namespace cv;

std::vector<int> calcGrayLevel(Mat &img) //计算灰度级，即算出从0到255区间的任意一个灰度值i，在图像img中有多少个像素点的灰度值为i
{
    assert(img.channels() == 1); //只计算灰度图像的
    std::vector<int> res(256);
    int rows = img.rows; //行
    int cols = img.cols; //列

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
        {
            int val = img.at<uchar>(i, j);
            res[val] += 1;
        }

    return res;
}

std::vector<double> pdf(std::vector<int> gray_level, Mat &img) //计算概率密度
{
    assert(gray_level.size() == 256);

    int N = img.rows * img.cols; //像素点总数
    std::vector<double> res(256);
    for (int i = 0; i < 256; ++i)
        res[i] = static_cast<double>(gray_level[i]) / N;

    return res;
}

std::vector<double> cdf(std::vector<double> pdf) //计算概率分布cdf
{
    assert(pdf.size() == 256);

    std::vector<double> res(256);
    res[0] = pdf[0];
    for (int i = 1; i < 256; ++i)
        res[i] = pdf[i] + res[i - 1];
    return res;
}

double findLow(double input_low, std::vector<double> cdf)
{
    assert(cdf.size() == 256);
    //找到分布概率大于我们的输入值input_low处最接近的灰度值，并以此作为最佳分割阈值的最小值
    for (int i = 0; i < 256; ++i)
        if (cdf[i] > input_low)
            // return cdf[i];
            return i;

    return 0.0;
}
double findHihg(double input_high, std::vector<double> cdf)
{
    assert(256 == cdf.size());
    //找到分布概率大于或等于我们的输入值input_high处最接近的灰度值，并以此作为最佳分割阈值的最大值
    for (int i = 0; i < 256; ++i)
        if (cdf[i] >= input_high)
            // return cdf[i];
            return i;

    return 0.0;
}

std::vector<double> strecthlim(Mat &img, double tol_low, double tol_high)
{
    std::vector<double> v(2);
    if (img.empty())
        return v;

    if (img.depth() == 0)
    {
        //计算灰度值
        std::vector<int> gray_level = calcGrayLevel(img);
        //计算概率密度pdf
        std::vector<double> p = pdf(gray_level, img);
        //计算概率分布cdf
        std::vector<double> c = cdf(p);
        //寻找tol_low, tol_high
        tol_low = findLow(tol_low, c);
        tol_high = findHihg(tol_high, c);

        if (tol_low == tol_high)   //该输入图像为等灰度值图像
            v = {0.0, 1.0};
        else
            v = {tol_low / 255.0, tol_high / 255.0};
    }
    else if (img.depth() == 5)
    {
        //计算灰度值
        std::vector<int> gray_level = calcGrayLevel_32FC(img);
        //计算概率密度pdf
        std::vector<double> p = pdf(gray_level, img);
        //计算概率分布cdf
        std::vector<double> c = cdf(p);
        //寻找tol_low, tol_high
        tol_low = findLow(tol_low, c);
        tol_high = findHihg(tol_high, c);

        if (tol_low == tol_high)
            v = {0.0, 1.0};
        else
            v = {tol_low, tol_high};
    }

    return v;
}

std::vector<int> calcGrayLevel_32FC(Mat &img) //计算灰度级，即算出从0到255区间的任意一个灰度值i，在图像img中有多少个像素点的灰度值为i
{
    assert(img.channels() == 1); //只计算灰度图像的
    std::vector<int> res(256);
    int rows = img.rows; //行
    int cols = img.cols; //列

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
        {
            int val = img.at<uchar>(i, j);
            res[val] += 1;
        }

    return res;
}

void test_function_strecthlim()
{
    Mat img = imread("./origin.jpg", 0);

    if (img.empty())
    {
        std::cout << "--------path of picture is uncorrect!------" << std::endl;
        return;
    }

    std::vector<double> v = strecthlim(img);
    for (auto &i : v)
        std::cout << i << std::endl;
}