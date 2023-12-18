#include "ImgAlgo.h"

using namespace std;
using namespace cv;


cv::Mat tyc_thread(cv::Mat &src){
    Mat res;
    Mat gray;
    cvtColor(src,gray,COLOR_BGR2GRAY);
    threshold(gray, res, 100, 255, THRESH_OTSU);
    return res;
}


int tyc_eight_TwoPassLabel(const Mat &src, Mat &labImg, Mat &flagImg)
{
    assert(src.type() == CV_8UC1);
    Mat bwImg = src.clone();
    bwImg = bwImg / 255;
    bwImg.convertTo(labImg, CV_32SC1);
    int rows = bwImg.rows;
    int cols = bwImg.cols;

    //二值图像像素值为0或1，为了不冲突，label从2开始
    int label = 2;
    //通过一个数组来设置并查集
    vector<int> labelSet;
    labelSet.push_back(0);
    labelSet.push_back(1);

    //第一次扫描
    int *data_prev;
    int *data_cur;
    int left, up, leftup, rightup;//指针指向的像素点的左方点和上方点,四邻域
    int neighborLabels[4];
    for (int i = 0; i < rows; i++)
    {
        if (i != 0) data_prev = labImg.ptr<int>(i - 1);
        else data_prev = nullptr;
        data_cur = labImg.ptr<int>(i);
        for (int j = 0; j < cols; j++)
        {
            if (data_cur[j] != 1)//当前点不为1，扫描下一个点
                continue;
            if(j==0) left = -1;
            else left = data_cur[j-1];

            if(j==0||i==0) leftup = -1;
            else leftup = data_prev[j-1];

            if(i==0) up = -1;
            else up = data_prev[j];

            if(i==0||j==cols-1) rightup = -1;
            else rightup = data_prev[j+1];

            int count = 0;
            for (int curLabel : {left, up, leftup, rightup})
            {
                //判断是否已经标记
                if (curLabel > 1)
                    neighborLabels[count++] = curLabel;
            }

            if (!count)//赋予一个新的label
            {
                labelSet.push_back(label);
                data_cur[j] = label;
                label++;
                continue;
            }
            //将当前点标记设为左点\上点\左上\右上点label的最小值
            sort(neighborLabels, neighborLabels + count);
            int smallestLabel = neighborLabels[0];
            data_cur[j] = smallestLabel;
            //设置并查集
            for (int k = 0; k < count; k++)
            {
                int neiLabel = neighborLabels[k];
                int oldSmallestLabel = labelSet[neiLabel];
                if (oldSmallestLabel > smallestLabel)
                {
                    while (oldSmallestLabel != neiLabel) {
                        int tmp = neiLabel;
                        neiLabel = oldSmallestLabel;
                        oldSmallestLabel = labelSet[neiLabel];
                        labelSet[tmp] = smallestLabel;
                    }
                    if (neiLabel > smallestLabel) {
                        labelSet[neiLabel] = smallestLabel;
                    }
                    else {
                        labelSet[smallestLabel] = neiLabel;
                        data_cur[j] = oldSmallestLabel;
                        k--;
                    }
                }
                else if (oldSmallestLabel < smallestLabel) {
                    labelSet[smallestLabel] = oldSmallestLabel;
                    data_cur[j] = oldSmallestLabel;
                }
            }
        }
    }

    int left2, up2, leftup2, rightup2;
    //8邻域需要第二次遍历
    int *data_prev2;
    for (int i = rows-1; i >=0; --i)
    {
        if(i!=0) data_prev2 = labImg.ptr<int>(i - 1);
        else data_prev2 = nullptr;
        if (i != rows-1) data_prev = labImg.ptr<int>(i + 1);
        else data_prev = nullptr;
        data_cur = labImg.ptr<int>(i);
        for (int j = cols-1; j >=0; j--)
        {
            if (data_cur[j] == 0)//当前点不为1，扫描下一个点
                continue;
            if (j == cols-1) left = -1;
            else left = data_cur[j + 1];

            if (j == cols-1 || i == rows-1) leftup = -1;
            else leftup = data_prev[j + 1];

            if (i == rows-1) up = -1;
            else up = data_prev[j];

            if (i == rows-1 || j == 0) rightup = -1;
            else rightup = data_prev[j - 1];

            //-------

            if (j == 0) left2 = -1;
            else left2 = data_cur[j - 1];

            if (j == 0 || i == 0) leftup2 = -1;
            else leftup2 = data_prev2[j - 1];

            if (i == 0) up2 = -1;
            else up2 = data_prev2[j];

            if (i == 0 || j == cols - 1) rightup2 = -1;
            else rightup2 = data_prev2[j + 1];


            int count = 0;
            for (int curLabel : {left, up, leftup, rightup,left2,up2,leftup2,rightup2})
            {
                //判断是否已经标记
                if (curLabel > 1)
                    neighborLabels[count++] = curLabel;
            }

            if (!count)//赋予一个新的label
            {
                labelSet.push_back(label);
                data_cur[j] = label;
                label++;
                continue;
            }
            //将当前点标记设为左点\上点\左上\右上点label的最小值
            sort(neighborLabels, neighborLabels + count);
            int smallestLabel = neighborLabels[0];
            data_cur[j] = smallestLabel;
            //设置并查集
            for (int k = 0; k < count; k++)
            {
                int neiLabel = neighborLabels[k];
                int oldSmallestLabel = labelSet[neiLabel];
                if (oldSmallestLabel > smallestLabel)
                {
                    while (oldSmallestLabel != neiLabel) {
                        int tmp = neiLabel;
                        neiLabel = oldSmallestLabel;
                        oldSmallestLabel = labelSet[neiLabel];
                        //labelSet[tmp] = smallestLabel;
                    }
                    if (neiLabel > smallestLabel) {
                        labelSet[neiLabel] = smallestLabel;
                    }
                    else {
                        labelSet[smallestLabel] = neiLabel;
                        data_cur[j] = oldSmallestLabel;
                    }
                }
                else if (oldSmallestLabel < smallestLabel) {
                    labelSet[smallestLabel] = oldSmallestLabel;
                    data_cur[j] = oldSmallestLabel;
                }
            }
        }
    }


    // 修改并查集
    for (int i = 0; i < (int)labelSet.size(); i++)
    {
        if (i == labelSet[i]) continue;
        int curLabel = labelSet[i];
        int prelabel = labelSet[curLabel];
        while (prelabel != curLabel)
        {
            curLabel = prelabel;
            prelabel = labelSet[prelabel];
        }
        labelSet[i] = curLabel;
    }
    for (int i = 0; i < (int)labelSet.size(); i++)
    {
        if (i == labelSet[i]) continue;
        int curLabel = labelSet[i];
        int prelabel = labelSet[curLabel];
        while (prelabel != curLabel)
        {
            curLabel = prelabel;
            prelabel = labelSet[prelabel];
        }
        labelSet[i] = curLabel;
    }

    unordered_map<int, int> um;
    int* datanew;
    int idxcc = 0;
    for (int i = 0; i < rows; i++)
    {
        datanew = labImg.ptr<int>(i);
        for (int j = 0; j < cols; j++) {
            datanew[j] = labelSet[datanew[j]];
            if (!um.count(datanew[j])) {
                um[datanew[j]] = idxcc;
                ++idxcc;
            }

        }
    }
    int retNum = um.size();
    RNG rng(10086);
    //设置绘图映射
    vector<Vec3b> colors(retNum);
    colors[0] = Vec3b(255, 255, 255);
    for (int i = 1; i < retNum; ++i) {
        Vec3b vec3 = Vec3b(rng.uniform(0, 254), rng.uniform(0, 254), rng.uniform(0, 254));
        colors[i] = (vec3);
    }
    flagImg = Mat::zeros(src.size(), CV_8UC3);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int tmp = labImg.at<int>(i, j);
            int idx = um[tmp];
            flagImg.at<Vec3b>(i, j) = colors[idx];
        }
    }
    return retNum;
}


int tyc_four_TwoPassLabel(const Mat &src, Mat &labImg, Mat &flagImg)
{
    assert(src.type() == CV_8UC1);
    Mat bwImg = src.clone();
    bwImg = bwImg / 255;
    bwImg.convertTo(labImg, CV_32SC1);
    int rows = bwImg.rows;
    int cols = bwImg.cols;

    //二值图像像素值为0或1，为了不冲突，label从2开始
    int label = 2;
    //通过一个数组来设置并查集
    vector<int> labelSet;
    labelSet.push_back(0);
    labelSet.push_back(1);

    //第一次扫描
    int *data_prev;
    int *data_cur;
    int left, up;//指针指向的像素点的左方点和上方点,四邻域
    int neighborLabels[2];
    for (int i = 0; i < rows; i++)
    {
        if (i != 0) data_prev = labImg.ptr<int>(i - 1);
        else data_prev = nullptr;
        data_cur = labImg.ptr<int>(i);
        for (int j = 0; j < cols; j++)
        {
            if (data_cur[j] != 1)//当前点不为1，扫描下一个点
                continue;
            if (j == 0) {
                left = -1;
            }
            else {
                left = data_cur[j - 1];
            }
            if (i == 0) {
                up = -1;
            }
            else {
                up = data_prev[j];
            }
            int count = 0;
            for (int curLabel : {left, up})
            {
                //判断是否已经标记
                if (curLabel > 1)
                    neighborLabels[count++] = curLabel;
            }

            if (!count)//赋予一个新的label
            {
                labelSet.push_back(label);
                data_cur[j] = label;
                label++;
                continue;
            }
            //将当前点标记设为左点和上点label的最小值
            sort(neighborLabels, neighborLabels + count);
            int smallestLabel = neighborLabels[0];
            //if (count == 2 && neighborLabels[1] < smallestLabel)
            //	smallestLabel = neighborLabels[1];
            data_cur[j] = smallestLabel;
            //设置并查集
            for (int k = 0; k < count; k++)
            {
                int neiLabel = neighborLabels[k];
                int oldSmallestLabel = labelSet[neiLabel];
                if (oldSmallestLabel > smallestLabel)
                {
                    while (oldSmallestLabel != neiLabel) {
                        int tmp = neiLabel;
                        neiLabel = oldSmallestLabel;
                        oldSmallestLabel = labelSet[neiLabel];
                        labelSet[tmp] = smallestLabel;
                    }
                    if (neiLabel > smallestLabel) {
                        labelSet[neiLabel] = smallestLabel;
                    }
                    else {
                        labelSet[smallestLabel] = neiLabel;
                        data_cur[j] = oldSmallestLabel;
                    }
                }
                else if (oldSmallestLabel < smallestLabel) {
                    labelSet[smallestLabel] = oldSmallestLabel;
                    data_cur[j] = oldSmallestLabel;
                }
            }
        }
    }
    // 修改并查集
    for (size_t i = 0; i < labelSet.size(); i++)
    {
        if (i == labelSet[i]) continue;
        int curLabel = labelSet[i];
        int prelabel = labelSet[curLabel];
        while (prelabel != curLabel)
        {
            curLabel = prelabel;
            prelabel = labelSet[prelabel];
        }
        labelSet[i] = curLabel;
    }
    //第二次扫描，用labelSet进行更新
    unordered_map<int, int> um;
    int* datanew;
    int idxcc = 0;
    for (int i = 0; i < rows; i++)
    {
        datanew = labImg.ptr<int>(i);
        for (int j = 0; j < cols; j++) {
            datanew[j] = labelSet[datanew[j]];
            if (!um.count(datanew[j])) {
                um[datanew[j]] = idxcc;
                ++idxcc;
            }

        }
    }
    int retNum = um.size();
    RNG rng(10086);
    //设置绘图映射
    vector<Vec3b> colors(retNum);
    colors[0] = Vec3b(255, 255, 255);
    for (int i = 1; i < retNum; ++i) {
        Vec3b vec3 = Vec3b(rng.uniform(0, 254), rng.uniform(0, 254), rng.uniform(0, 254));
        colors[i] = (vec3);
    }
    flagImg = Mat::zeros(src.size(), CV_8UC3);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int tmp = labImg.at<int>(i, j);
            int idx = um[tmp];
            flagImg.at<Vec3b>(i, j) = colors[idx];
        }
    }
    return retNum;
}



//输入的是大津法二值化的结果
int tyc_CountConnectComponent(cv::Mat &src, cv::Mat &result,int method){
    assert(src.channels()==1);
    if(method==4){
        Mat labImg;
        int retnum = tyc_four_TwoPassLabel(src,labImg,result);
        return retnum;
    }
    else{
        Mat labImg;
        int retnum = tyc_eight_TwoPassLabel(src,labImg,result);
        return retnum;
    }
    return 0;
}

vector<int> CalHistgramOneChannel(const Mat& img) {
    assert(img.channels() == 1);
    vector<int> histogram(256, 0);
    for (int row = 0; row < img.rows; row++)
    {
        for (int col = 0; col < img.cols; col++)
        {
            auto idx = img.at<uchar>(row, col);
            histogram[idx]++;
        }
    }
    return histogram;
}


void tyc_EquHistogram(const Mat& src, Mat& out) {
    assert(src.channels()==1);
    assert(src.cols == out.cols && src.rows == out.rows);
    vector<int> histogram = CalHistgramOneChannel(src);
    //建立映射
    vector<double>map(256, 0);

    double total = src.cols * src.rows;

    for (int i = 0; i < (int)histogram.size(); ++i) {
        map[i] = (double)(i == 0 ? 0 : map[i - 1]) + (double)histogram[i] * 255.0 / total;
    }
    map[255] = 1.0;
    //均衡化
    for (int row = 0; row < src.rows; row++)
    {
        for (int col = 0; col < src.cols; col++)
        {
            auto tmp = src.at<uchar>(row, col);
            out.at<uchar>(row, col) = saturate_cast<uchar>(map[tmp]);
        }
    }
}


// 绘制简易直方图
cv::Mat DrawHistImg(cv::Mat &src)
{
    cv::Mat hist = cv::Mat::zeros(1, 256, CV_32FC1);
    for (int i = 0; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
            hist.at<float>(0, src.at <uchar>(i, j))++;
        }
    }
    cv::Mat histImage = cv::Mat::zeros(500, 960, CV_8UC3);
    const int bins = 255;
    double maxValue;
    cv::Point2i maxLoc;
    cv::minMaxLoc(hist, 0, &maxValue, 0, &maxLoc);
    int scale = 4;
    int histHeight = 500;

    for (int i = 0; i < bins; i++)
    {
        float binValue = (hist.at<float>(i));
        int height = cvRound(binValue * histHeight / maxValue);
        cv::rectangle(histImage, cv::Point(i * scale, histHeight),
            cv::Point((i + 1) * scale - 1, histHeight - height), cv::Scalar(255, 0, 255), -1);

    }
    return histImage;
}


/**
 一维高斯卷积，对每行进行高斯卷积
 img 输入原图像
 dst  一维高斯卷积后的输出图像
 */
void gaussianConvolution(Mat &img, Mat &dst) {
    int nr = img.rows;
    int nc = img.cols;
    int templates[3] = { 1, 2, 1 };

    // 按行遍历除每行边缘点的所有点
    for (int j = 0; j < nr; j++) {
        uchar* data = img.ptr<uchar>(j); //提取该行地址
        for (int i = 1; i < nc - 1; i++) {
            int sum = 0;
            for (int n = 0; n < 3; n++) {
                sum += data[i - 1 + n] * templates[n]; //相称累加
            }
            sum /= 4;
            dst.ptr<uchar>(j)[i] = sum;
        }
    }
}


/*
 高斯滤波器，利用3*3的高斯模版进行高斯卷积
 img 输入原图像
 dst  高斯滤波后的输出图像
*/
void gaussianFilter(Mat &img, Mat &dst) {
    // 对水平方向进行滤波
    Mat dst1 = img.clone();
    gaussianConvolution(img, dst1);
    //图像矩阵转置
    Mat dst2;
    transpose(dst1, dst2);
    // 对垂直方向进行滤波
    Mat dst3 = dst2.clone();
    gaussianConvolution(dst2, dst3);
    // 再次转置
    transpose(dst3, dst);
}


/*
 用一阶偏导有限差分计算梯度幅值和方向
 img 输入原图像
 gradXY 输出的梯度幅值
 theta 输出的梯度方向
 */
void getGrandient(Mat &img, Mat &gradXY, Mat &theta) {
    gradXY = Mat::zeros(img.size(), CV_8U);
    theta = Mat::zeros(img.size(), CV_8U);

    for (int j = 1; j < img.rows - 1; j++) {
        for (int i = 1; i < img.cols - 1; i++) {
            double gradY = double(img.ptr<uchar>(j - 1)[i - 1] + 2 * img.ptr<uchar>(j - 1)[i] + img.ptr<uchar>(j - 1)[i + 1] - img.ptr<uchar>(j + 1)[i - 1] - 2 * img.ptr<uchar>(j + 1)[i] - img.ptr<uchar>(j + 1)[i + 1]);
            double gradX = double(img.ptr<uchar>(j - 1)[i + 1] + 2 * img.ptr<uchar>(j)[i + 1] + img.ptr<uchar>(j + 1)[i + 1] - img.ptr<uchar>(j - 1)[i - 1] - 2 * img.ptr<uchar>(j)[i - 1] - img.ptr<uchar>(j + 1)[i - 1]);

            gradXY.ptr<uchar>(j)[i] = sqrt(gradX*gradX + gradY * gradY); //计算梯度
            theta.ptr<uchar>(j)[i] = atan(gradY / gradX); //计算梯度方向
        }
    }
}


/*
 局部非极大值抑制
 gradXY 输入的梯度幅值
 theta 输入的梯度方向
 dst 输出的经局部非极大值抑制后的图像
 */
void nonLocalMaxValue(Mat &gradXY, Mat &theta, Mat &dst) {
    dst = gradXY.clone();
    for (int j = 1; j < gradXY.rows - 1; j++) {
        for (int i = 1; i < gradXY.cols - 1; i++) {
            double t = double(theta.ptr<uchar>(j)[i]);
            double g = double(dst.ptr<uchar>(j)[i]);
            if (g == 0.0) {
                continue;
            }
            double g0, g1;
            if ((t >= -(3 * CV_PI / 8)) && (t < -(CV_PI / 8))) {
                g0 = double(dst.ptr<uchar>(j - 1)[i - 1]);
                g1 = double(dst.ptr<uchar>(j + 1)[i + 1]);
            }
            else if ((t >= -(CV_PI / 8)) && (t < CV_PI / 8)) {
                g0 = double(dst.ptr<uchar>(j)[i - 1]);
                g1 = double(dst.ptr<uchar>(j)[i + 1]);
            }
            else if ((t >= CV_PI / 8) && (t < 3 * CV_PI / 8)) {
                g0 = double(dst.ptr<uchar>(j - 1)[i + 1]);
                g1 = double(dst.ptr<uchar>(j + 1)[i - 1]);
            }
            else {
                g0 = double(dst.ptr<uchar>(j - 1)[i]);
                g1 = double(dst.ptr<uchar>(j + 1)[i]);
            }

            if (g <= g0 || g <= g1) {
                dst.ptr<uchar>(j)[i] = 0.0;
            }
        }
    }
}

/*
 弱边缘点补充连接强边缘点
 img 弱边缘点补充连接强边缘点的输入和输出图像
 */
void doubleThresholdLink(Mat &img) {
    // 循环找到强边缘点，把其领域内的弱边缘点变为强边缘点
    for (int j = 1; j < img.rows - 2; j++) {
        for (int i = 1; i < img.cols - 2; i++) {
            // 如果该点是强边缘点
            if (img.ptr<uchar>(j)[i] == 255) {
                // 遍历该强边缘点领域
                for (int m = -1; m < 1; m++) {
                    for (int n = -1; n < 1; n++) {
                        // 该点为弱边缘点（不是强边缘点，也不是被抑制的0点）
                        if (img.ptr<uchar>(j + m)[i + n] != 0 && img.ptr<uchar>(j + m)[i + n] != 255) {
                            img.ptr<uchar>(j + m)[i + n] = 255; //该弱边缘点补充为强边缘点
                        }
                    }
                }
            }
        }
    }

    for (int j = 0; j < img.rows - 1; j++) {
        for (int i = 0; i < img.cols - 1; i++) {
            // 如果该点依旧是弱边缘点，及此点是孤立边缘点
            if (img.ptr<uchar>(j)[i] != 255 && img.ptr<uchar>(j)[i] != 255) {
                img.ptr<uchar>(j)[i] = 0; //该孤立弱边缘点抑制
            }
        }
    }
}


/*
 用双阈值算法检测和连接边缘
 low 输入的低阈值
 high 输入的高阈值
 img 输入的原图像
 dst 输出的用双阈值算法检测和连接边缘后的图像
 */
void doubleThreshold(double low, double high, Mat &img, Mat &dst) {
    dst = img.clone();

    // 区分出弱边缘点和强边缘点
    for (int j = 0; j < img.rows - 1; j++) {
        for (int i = 0; i < img.cols - 1; i++) {
            double x = double(dst.ptr<uchar>(j)[i]);
            // 像素点为强边缘点，置255
            if (x > high) {
                dst.ptr<uchar>(j)[i] = 255;
            }
            // 像素点置0，被抑制掉
            else if (x < low) {
                dst.ptr<uchar>(j)[i] = 0;
            }
        }
    }

    // 弱边缘点补充连接强边缘点
    doubleThresholdLink(dst);
}


// 输入灰度图,输出Canny算子
void tyc_Canny(cv::Mat &img, cv::Mat &dstCanny) {
    assert(img.channels() == 1);
    //高斯滤波器
    Mat gauss_img;
    gaussianFilter(img, gauss_img);

    // 用一阶偏导有限差分计算梯度幅值和方向,sobel算子
    Mat gradXY, theta;
    getGrandient(gauss_img, gradXY, theta);

    // 局部非极大值抑制
    Mat local_img;
    nonLocalMaxValue(gradXY, theta, local_img);

    // 用双阈值算法检测和连接边缘
    Mat dst;
    doubleThreshold(50, 100, local_img, dst);
    dstCanny = dst.clone();
}


void calCenterHuidu(int idx_x, int idx_y, int lenx, int leny, vector<pair<double, double>> &ret,
    Mat &src, Mat &thres, int target) {
    int rowbegin = idx_y;
    int rowend = idx_y + leny;
    int colbegin = idx_x;
    int colend = idx_x + lenx;
    double sumx = 0.0;
    double sumy = 0.0;
    double retx = 0.0, rety = 0.0;
    uchar* tmp;
    int* bineay;
    for (int i = rowbegin; i <= rowend; ++i) {
        bineay = thres.ptr<int>(i);
        tmp = src.ptr<uchar>(i);
        for (int j = colbegin; j <= colend; ++j) {
            if (bineay[j] == target) {
                sumx += 255 - tmp[j];
            }
        }
    }
    sumy = sumx;
    for (int i = rowbegin; i <= rowend; ++i) {
        tmp = src.ptr<uchar>(i);
        bineay = thres.ptr<int>(i);
        for (int j = colbegin; j <= colend; ++j) {
            if (bineay[j] == target) {
                double incx = j * (255 - tmp[j]);
                incx /= sumx;
                retx += incx;
                double incy = i * (255 - tmp[j]);
                incy /= sumy;
                rety += incy;
            }
        }
    }
    ret[target].first = retx;
    ret[target].second = rety;
}


void makeBoard(Mat &src, Mat &dst, unordered_set<int> &um) {
    int rows = src.rows;
    int cols = src.cols;
    uchar* tmp;
    int* bineay;
    for (int i = 0; i < rows; ++i) {
        bineay = src.ptr<int>(i);
        tmp = dst.ptr<uchar>(i);
        for (int j = 0; j < cols; ++j) {
            int idx = bineay[j];
            if (!um.count(idx)) {
                tmp[j] = 0;
            }
        }
    }
}


std::vector<std::pair<double,double>> tyc_det_ciclePos(cv::Mat&img,cv::Mat &res,cv::Mat& background
                                                       ,cv::Mat &connectedImg){

    Mat srcshow = img.clone();
    cvtColor(img, img, COLOR_BGR2GRAY);

    //均值滤波
    //调用均值滤波函数blur()进行滤波
    int rows = img.rows;
    int cols = img.cols;
    //根据图像设置卷积核
    int kernel_size = min(rows / 230, cols / 230);


    Mat blurImg;
    blur(img, blurImg, Size(kernel_size, kernel_size));

    //大津法二值化
    Mat OtsuImg;
    threshold(blurImg, OtsuImg, 100, 255, THRESH_OTSU);  //大津法

    //形态学操作，闭区间填补空洞
    Mat element;
    element = getStructuringElement(MORPH_RECT, Size(kernel_size * 3, kernel_size * 3));
    Mat element2, DilateImg;
    element2 = getStructuringElement(MORPH_ELLIPSE, Size(kernel_size, kernel_size));
    //膨胀删除原点
    morphologyEx(OtsuImg, DilateImg, MORPH_DILATE, element);

    //连通域分析得出最大的连通域
    Mat out, status, centroids;
    int number = connectedComponentsWithStats(DilateImg, out, status, centroids, 8, CV_32S);
    int targetCount = -1;
    int areaTmp = -1;
    int *ptarea;
    for (int i = 1; i < status.rows; ++i) {
        ptarea = status.ptr<int>(i);
        if (areaTmp < ptarea[4]) {
            areaTmp = ptarea[4];
            targetCount = i;
        }
    }
    assert(targetCount != -1);
    //做交集，修改滤波图像，删除背景
    int* pt1;
    uchar* pt2;
    for (int i = 0; i < rows; ++i) {
        pt1 = out.ptr<int>(i);
        pt2 = blurImg.ptr<uchar>(i);
        for (int j = 0; j < cols; ++j) {
            if (pt1[j] != targetCount) {
                pt2[j] = 0;
            }
        }
    }
    //重新大津法,同时反色
    threshold(blurImg, OtsuImg, 100, 255, THRESH_BINARY_INV+THRESH_OTSU);  //大津法

    //因为是黑色，所以用开操作填补空洞
    Mat CloseImg;
    morphologyEx(OtsuImg, CloseImg, MORPH_CLOSE, element2);

    //进一步连通域分析，灰度重心法求取质心，判断是否在背景框捏，得出中心点坐标
    Mat outlast, statuslast, centroidslast;
    long long allarea = rows / 230 * cols;
    int number_last = connectedComponentsWithStats(CloseImg, outlast, statuslast, centroidslast, 8, CV_32S);

    RNG rng(10086);
    //设置绘图映射
    vector<Vec3b> colors(number_last);
    colors[0] = Vec3b(255, 255, 255);
    for (int i = 1; i < number_last; ++i) {
        Vec3b vec3 = Vec3b(rng.uniform(0, 254), rng.uniform(0, 254), rng.uniform(0, 254));
        colors[i] = (vec3);
    }

//    vector<pair<double, double>> retPoints;//x是列，y是行
//    for (int i = 1; i < number_last; ++i) {
//        ptarea = statuslast.ptr<int>(i);
//        if (ptarea[4] > allarea) continue;

//        // 中心位置
//        retPoints.emplace_back(pair<double, double>{centroidslast.at<double>(i, 0), centroidslast.at<double>(i, 1)});
//        int center_x = static_cast<int>(centroidslast.at<double>(i, 0) + 0.5);
//        int center_y = static_cast<int>(centroidslast.at<double>(i, 1) + 0.5);
//        circle(srcshow, Point(center_x, center_y), 3, colors[i], -1, 8, 0);
//    }

    // 灰度重心法计算中心点
    unordered_set<int> um_flag;
    vector<pair<double, double>> retPointstmp(number_last, pair<double, double>{-1, -1});//x是列，y是行
    for (int i = 1; i < number_last; ++i) {
        ptarea = statuslast.ptr<int>(i);
        if (ptarea[4] > allarea) continue;
        um_flag.insert(i);
        int target = i;
        int idxx = ptarea[0];
        int idxy = ptarea[1];
        int xlen = ptarea[2];
        int ylen = ptarea[3];
        calCenterHuidu(idxx, idxy, xlen, ylen, retPointstmp, img, outlast, target);
    }
    Mat showbacktmp = CloseImg.clone();
    makeBoard(outlast, showbacktmp, um_flag);


    vector<pair<double, double>> retPoints;//x是列，y是行
    //for (int i = 1; i < number_last; ++i) {
    //	ptarea = statuslast.ptr<int>(i);
    //	if (ptarea[4] > allarea) continue;

    //	// 中心位置
    //	retPoints.emplace_back(pair<double, double>{centroidslast.at<double>(i, 0), centroidslast.at<double>(i, 1)});
    //	int center_x = static_cast<int>(centroidslast.at<double>(i, 0) + 0.5);
    //	int center_y = static_cast<int>(centroidslast.at<double>(i, 1) + 0.5);
    //	circle(srcshow, Point(center_x, center_y), 3, colors[i], -1, 8, 0);
    //}
    for (int i = 0; i < (int)retPointstmp.size(); ++i) {
        if (retPointstmp[i].first > 0 && retPointstmp[i].second > 0) {
            // 中心位置
            retPoints.emplace_back(retPointstmp[i]);
            int center_x = static_cast<int>(retPointstmp[i].first + 0.5);
            int center_y = static_cast<int>(retPointstmp[i].second + 0.5);
            circle(srcshow, Point(center_x, center_y), 3, colors[i], -1, 8, 0);
        }
    }

    res = srcshow.clone();
    background = DilateImg.clone();
    connectedImg = showbacktmp.clone();
    return retPoints;
}


void tyc_resize(cv::Mat &src, int hei, int wid){
    int rows = src.rows;//hei
    int cols = src.cols;//wid
    double rbei = (double)rows/(double)hei;
    double cbei = (double)cols/(double)wid;
    double maxscale = max(rbei,cbei);
    int newrow = static_cast<int>((double)rows/maxscale);
    int newcol = static_cast<int>((double)cols/maxscale);
    resize(src, src, cv::Size(newcol, newrow), 0, 0, INTER_LINEAR);
}





























