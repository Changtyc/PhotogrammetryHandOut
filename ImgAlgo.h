#ifndef IMGALGO_H
#define IMGALGO_H
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <opencv2/opencv.hpp>

//二值化
cv::Mat tyc_thread(cv::Mat &src);

//连通域计算
//输入的是二值化的图
int tyc_CountConnectComponent(cv::Mat &src, cv::Mat &result,int method);

int tyc_eight_TwoPassLabel(const cv::Mat &src, cv::Mat &labImg, cv::Mat &flagImg);

int tyc_four_TwoPassLabel(const cv::Mat &src, cv::Mat &labImg, cv::Mat &flagImg);

//计算单通道直方图
std::vector<int> CalHistgramOneChannel(const cv::Mat& img);

//直方图均衡化
void tyc_EquHistogram(const cv::Mat& src, cv::Mat& out);

//绘制直方图
cv::Mat DrawHistImg(cv::Mat &src);


//Canny算子相关函数
//高斯滤波器
void gaussianConvolution(cv::Mat &img, cv::Mat &dst);
void gaussianFilter(cv::Mat &img, cv::Mat &dst);

//基于sobel算子计算梯度和法向
void getGrandient(cv::Mat &img, cv::Mat &gradXY, cv::Mat &theta);

//非极大值抑制
void nonLocalMaxValue(cv::Mat &gradXY, cv::Mat &theta, cv::Mat &dst);

//弱边缘点补充连接强边缘点
void doubleThresholdLink(cv::Mat &img);

//用双阈值算法检测和连接边缘
void doubleThreshold(double low, double high, cv::Mat &img, cv::Mat &dst);

//自定义Canny算子
void tyc_Canny(cv::Mat &img, cv::Mat &dstCanny);

//中心点检测
std::vector<std::pair<double,double>> tyc_det_ciclePos(cv::Mat&img,cv::Mat &res,cv::Mat& background
                                                       ,cv::Mat &connectedImg);
//缩放图片
void tyc_resize(cv::Mat &src, int hei, int wid);

#endif // IMGALGO_H
