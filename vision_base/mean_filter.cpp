// https://blog.csdn.net/weixin_40647819/article/details/92719036 数字图像处理基础算法
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void MeanFilater(cv::Mat& src,cv::Mat& dst,cv::Size wsize){
	//图像边界扩充:窗口的半径
	if (wsize.height % 2 == 0 || wsize.width % 2 == 0){
		fprintf(stderr,"Please enter odd size!" );
		exit(-1);
	}
	int hh = (wsize.height - 1) / 2;
	int hw = (wsize.width - 1) / 2;
	cv::Mat Newsrc; //构造图像
    

	cv::copyMakeBorder(src, Newsrc, hh, hh, hw, hw, cv::BORDER_REFLECT_101);//以边缘为轴，对称
	dst=cv::Mat::zeros(src.size(),src.type());
 
    //均值滤波
	int sum = 0;
	int mean = 0;
	for (int i = hh; i < src.rows + hh; ++i){
		for (int j = hw; j < src.cols + hw;++j){
 
			for (int r = i - hh; r <= i + hh; ++r){
				for (int c = j - hw; c <= j + hw;++c){
					sum = Newsrc.at<uchar>(r, c) + sum; //at()像素操作,表示某个通道中(r,c)位置的像素值
				}
			}
			mean = sum / (wsize.area()); //template<typename _Tp> inline _Tp Size_<_Tp>::area() const { return width*height; }
			dst.at<uchar>(i-hh,j-hw)=mean;
			sum = 0;
			mean = 0;
		}
	}
 
}
 
int main(){
	cv::Mat src = cv::imread("test.jpg");
	if (src.empty()){
		return -1;
	}
	if (src.channels() > 1)
		cv::cvtColor(src,src,CV_RGB2GRAY);
 
	cv::Mat dst;
	cv::Mat dst1;
	cv::Size wsize(7,7);
 
	double t2 = (double)cv::getTickCount();
	MeanFilater(src, dst, wsize); //均值滤波
	t2 = (double)cv::getTickCount() - t2;
	double time2 = (t2 *1000.) / ((double)cv::getTickFrequency());
	std::cout << "FASTmy_process=" << time2 << " ms. " << std::endl << std::endl;
 
	cv::namedWindow("src");
	cv::imshow("src", src);
	cv::namedWindow("dst");
	cv::imshow("dst", dst);
	cv::imwrite("mean_filter.jpg",dst);
	cv::waitKey(0);
}