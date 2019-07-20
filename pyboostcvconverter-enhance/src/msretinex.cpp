
#include <iostream>
#include <opencv2/opencv.hpp>
#include "msretinex.h"
#include "stretchlim.h"
#include "imadjust.h"

using namespace std;
using namespace cv;

Mat MSRCR(const Mat &I)
{
    int lowScale = 15, medScale = 80, highScale = 250, leftChop = 1, rightChop = 1;
    Mat dst;
    I.convertTo(dst, CV_32FC3, 1, 1);

    Mat msr = MSR(dst, lowScale, medScale, highScale);
    // PresentFirstrow(msr);

    //需要对RGB进行分离
    vector<Mat> chan_BGR, chan_BGR_MSR;
    split(dst, chan_BGR);
    split(msr, chan_BGR_MSR);
    Mat siAdd = chan_BGR[0] + chan_BGR[1] + chan_BGR[2];
    // cout << "channels of siAdd = " << siAdd.channels() << endl;
    MatLog(siAdd);

    //测试
    // vector< vector<double>> vecdd;
    // vector<double> v1(2),v2(2),v3(2);
    // v1 = {0.4082, 0.8648};
    // v2 = {0.5342, 0.8797};
    // v3 = {0.5704, 0.8936};
    // vecdd.push_back(v1);
    // vecdd.push_back(v2);
    // vecdd.push_back(v3);
    vector<Mat> outrgb;
    for (int c = 0; c < 3; c++)
    {
        Mat temp0 = chan_BGR[c].clone();
        MatMultiNum(temp0, 125);
        MatLog(temp0);
        Mat cr = temp0 - siAdd;
        // PresentFirstrow(cr);

        MatMul(cr, chan_BGR_MSR[c]);
        normalize(cr, cr, 0, 1, NORM_MINMAX);
        Mat vmat;
        cr.convertTo(vmat, CV_8UC1, 255, 1);
        vector<double> v = strecthlim(vmat);
        // Mat sigMat;
        // cr.convertTo(sigMat, CV_8UC1, 255, 0);
        // imadjust(sigMat, sigMat, vecdd[c], 0, 1, 1);
        Mat sigMat = (cr - v[0]);
        sigMat = sigMat / (v[1] - v[0]);
        outrgb.push_back(sigMat);
    }
    //对RGB进行合并
    Mat out;
    merge(outrgb, out);
    // PresentFirstrow(out);
    return out;
}

Mat MSR(const Mat &I, int lowScale, int medScale, int highScale)
{
    Mat dst;
    I.convertTo(dst, I.type(), 255, 1);
    dst.convertTo(dst, dst.type(), 255, 1);
    int scale[3] = {lowScale, medScale, highScale};
    Mat out(dst.size(), dst.type(), Scalar{0, 0, 0});
    for (int i = 0; i < 3; i++)
    {
        out = out + (1.0 / 3.0 * SSR(dst, scale[i]));
    }
    return out;
}

Mat SSR(const Mat &I, int arg)
{
    Size size = {arg * 4 + 1, arg * 4 + 1};
    Mat temp;
    GaussianBlur(I, temp, size, arg, arg, BORDER_REPLICATE);
    Mat out1 = I.clone();

    Mat out2 = MatAddNum(temp, 1.0);
    ;
    MatLog(out1);
    MatLog(out2);
    Mat out = out1 - out2;
    out = MatAddNum(out, 0.5);
    return out;
}

Mat multiscaleRetinex(const Mat &matin)
{
    Mat out;
    out = MSRCR(matin);
    out.convertTo(out, CV_8UC3, 255);
    return out;
}

void MatLog(Mat &m)
{
    // cout << "m.size() = " << m.size() << ", m.type() = " << m.type() << endl;
    int channel = m.channels();
    for (int row = 0; row < m.rows; row++)
    {
        float *data = (float *)m.ptr<float>(row);
        for (int col = 0; col < m.cols; col++)
        {
            for (int i = 0; i < channel; i++)
            {
                data[col * channel + i] = log(data[col * channel + i]);
            }
        }
    }
}

void PresentFirstrow(const Mat &m)
{
    cout << "m.size = " << m.size() << "; m.type = " << m.type() << endl;
    Mat dst;
    if (m.type() == CV_8UC3 || m.type() == CV_8UC1)
    {
        if (m.type() == CV_8UC3)
        {
            m.convertTo(dst, CV_32FC3);
        }
        else if (m.type() == CV_8UC1)
        {
            m.convertTo(dst, CV_32FC1);
        }
        else
        {
            return;
        }
    }
    else if (m.type() == CV_32FC3 || m.type() == CV_32FC1)
    {
        dst = m.clone();
    }
    else
    {
        cout << "PresentFirstrow error!" << endl;
        return;
    }
    int row = 400; //特殊值，测试用
    float *data = (float *)dst.ptr<float>(row);
    int clos = 6; //特殊值，测试用
    int channel = m.channels();
    int col = 800;
    cout << "the " << row << " row " << col << " col of Mat is:" << endl;
    for (; col < 800 + clos; col++) //特殊值，测试用
    {
        cout << "[";
        for (int i = 0; i < channel; i++)
        {
            cout << data[col * channel + i] << " ";
        }
        cout << "]\n"
             << endl;
    }
}

Mat MatAddNum(const Mat &m, double num)
{
    Mat out;
    int depth = m.depth();
    int channel = m.channels();
    if (depth == 0)
    {
        unsigned char i = num;
        if (channel == 1)
        {
            out = Mat(m.rows, m.cols, CV_8UC1, Scalar{i});
        }
        else if (channel == 3)
        {
            out = Mat(m.rows, m.cols, CV_8UC3, Scalar{i, i, i});
        }
        else
        {
            return out;
        }
        out = out + m;
    }
    else if (depth == 5)
    {
        float f = num;
        if (channel == 1)
        {
            out = Mat(m.rows, m.cols, CV_32FC1, Scalar{f});
        }
        else if (channel == 3)
        {
            out = Mat(m.rows, m.cols, CV_32FC3, Scalar{f, f, f});
        }
        else
        {
            return out;
        }
        out = out + m;
    }
    else
    {
        return out;
    }

    return out;
}

void MatMultiNum(Mat &m, double num)
{
    // cout << "m.size() = " << m.size() << ", m.type() = " << m.type() << endl;
    int channel = m.channels();
    for (int row = 0; row < m.rows; row++)
    {
        float *data = (float *)m.ptr<float>(row);
        for (int col = 0; col < m.cols; col++)
        {
            for (int i = 0; i < channel; i++)
            {
                data[col * channel + i] = num * (data[col * channel + i]);
            }
        }
    }
}

void MatMul(Mat &a, const Mat &b)
{
    int channel = a.channels();
    for (int row = 0; row < a.rows; row++)
    {
        float *data1 = (float *)a.ptr<float>(row);
        float *data2 = (float *)b.ptr<float>(row);
        for (int col = 0; col < a.cols; col++)
        {
            for (int i = 0; i < channel; i++)
            {
                data1[col * channel + i] = data1[col * channel + i] * data2[col * channel + i];
            }
        }
    }
}

Mat GetSingleChannel(const Mat &img, int ch)
{
    Mat bgr(img.rows, img.cols, CV_8UC3, Scalar(0, 0, 0));
    Mat out[] = {bgr};
    int from_to[] = {ch, ch};
    mixChannels(&img, 1, out, 1, from_to, 1);
    //获得其中一个通道的数据进行分析
    // imshow("1 channel", bgr);
    // waitKey();
    return bgr;
}