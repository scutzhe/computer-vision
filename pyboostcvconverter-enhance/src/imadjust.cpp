//imadjust函数实现

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <vector>

#include "imadjust.h"
using namespace cv;
using namespace std;

void imadjust(Mat &input, Mat &output, double low_in, double high_in, double low_out, double high_out, double gamma)
{
    assert(low_in < high_in && is0to1(low_in) && is0to1(high_in) && is0to1(low_out) && is0to1(high_out));

    //将matlab中的灰度值区间[0,1]转为opencv灰度值区间[0,255]
    high_in *= 255;
    high_out *= 255;
    low_in *= 255;
    low_out *= 255;

    imadjust2(input, output, low_in, high_in, low_out, high_out, gamma);
}
void imadjust(Mat &input, Mat &output, vector<double> in, double low_out, double high_out, double gamma)
{
    assert(2 == in.size());
    double low_in = in[0];
    double high_in = in[1];
    imadjust(input, output, low_in, high_in, low_out, high_out, gamma);
}

void imadjust2(Mat &input, Mat &output, int low_in, int high_in, int low_out, int high_out, double gamma) //opencv，像素区间[0,255]
{
    output = input.clone();
    int rows = input.rows; //行
    int cols = input.cols; //列
    double k = (static_cast<double>(high_out) - low_out) / (high_in - low_in);
    vector<uchar> gamma_lut = gammaLut(gamma);

    switch (input.channels())
    {

    case 1: //灰度图

        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
            {
                double result = 0;
                if (input.at<uchar>(i, j) <= low_in) //灰度值小于low_in的像素点
                {
                    result = low_out; //结果为low_out
                }
                else if (low_in < input.at<uchar>(i, j) && input.at<uchar>(i, j) < high_in) //灰度值在[low_in, high_in]
                {
                    result = k * (input.at<uchar>(i, j) - low_in) + high_in; //灰度值线性变换
                    result = gamma_lut[static_cast<uchar>(result)];          //灰度值gamma变换
                }
                else
                {
                    result = high_out; //灰度值大于high_in的像素点，结果为high_out
                }

                output.at<uchar>(i, j) = static_cast<uchar>(result) % 255;
            }
        break;

        //彩色图片
    case 3:
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                for (int k = 0; k < 3; ++k)
                {
                    double result = 0;
                    if (input.at<Vec3b>(i, j)[k] <= low_in)
                        result = low_out;
                    else if (low_in < input.at<Vec3b>(i, j)[k] && input.at<Vec3b>(i, j)[k] < high_in)
                    {
                        result = k * (input.at<Vec3b>(i, j)[k] - low_in) + high_in;
                        result = gamma_lut[static_cast<uchar>(result)];
                    }
                    else
                    {
                        result = high_out;
                    }

                    output.at<Vec3b>(i, j)[k] = static_cast<uchar>(result) % 255;
                }
        break;

    default:
        break;
    }
}

bool is0to1(const double var)
{
    return 0 <= var && var <= 1;
}

vector<uchar> gammaLut(const double gamma, const double c)
{
    vector<uchar> lut(256);
    for (int i = 0; i < 256; ++i)
        lut[i] = static_cast<uchar>(c * pow((double)(i / 255.0), gamma) * 255.0);

    return lut;
}

void test_function_imadjust()
{
    Mat src_img = imread("./origin.jpg");
    if (src_img.empty())
    {
        cout << "--------path of picture is uncorrect!------" << endl;
        return;
    }

    Mat dst_img;
    imadjust(src_img, dst_img, 0, 1, 0, 1, 2);

    imshow("src_img", src_img);

    imshow("dst_img", dst_img);
    waitKey(0);
}