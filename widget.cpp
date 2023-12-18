#include "widget.h"
#include "ui_widget.h"
#include <QMessageBox>
#include <QFileDialog>
#include <QDateTime>
#include <QtDebug>
#include <QImage>


Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);
    ui->lineEdit->setText("8");
}

Widget::~Widget()
{
    delete ui;
}



void Widget::on_pushButton_input1_clicked()
{
    using namespace cv;
    QString open_file, open_file_path;
    open_file = QFileDialog::getOpenFileName(this,
                                             "please choose an image file for component",
                                             QDir::currentPath(),
                                             "Image Files(*.jpg *.png *.bmp);;All(*.*)");
    if(open_file.isNull()){
        QMessageBox::warning(this, tr("read image"), tr("failly read image!"));
        return;
    }
    qDebug()<<open_file<<endl;
    m_srcimgPath = open_file;
    Mat mat = imread(open_file.toStdString());
    tyc_resize(mat,LabelHeight3,LabelWigth3);
    cvtColor(mat,mat,COLOR_BGR2RGB);
    QImage Result;
    Result = QImage((const uchar*)(mat.data), mat.cols, mat.rows,
                    mat.cols*mat.channels(),QImage::Format_RGB888);
    ui->label_input->setAlignment(Qt::AlignCenter);
    ui->label_input->setPixmap(QPixmap::fromImage(Result));
    QMessageBox::warning(this, tr("read image"), tr("successfully read image!"));
}

//处理连通成分
void Widget::on_pushButton_deal1_clicked()
{
    int idx = ui->comboBox->currentIndex();
    if(!m_srcimgPath.size()){
        QMessageBox::warning(this, tr("connected component"), tr("please input image!"));
        return;
    }
    // 二值化图像
    using namespace cv;
    Mat thresSrc;
    Mat src;
    src = imread(m_srcimgPath.toStdString());
    thresSrc = tyc_thread(src);
    //显示二值化图像
    Mat thresshow = thresSrc.clone();
    tyc_resize(thresshow,LabelHeight1,LabelWigth1);
    cvtColor(thresshow,thresshow,COLOR_GRAY2RGB);
    QImage Result;
    Result = QImage((const uchar*)(thresshow.data), thresshow.cols, thresshow.rows,
                    thresshow.cols*thresshow.channels(),QImage::Format_RGB888);
    ui->label_thread->setAlignment(Qt::AlignCenter);
    ui->label_thread->setPixmap(QPixmap::fromImage(Result));

    //连通域分析
    Mat result;
    int count = 0;
    if(idx){
        Mat tmp;
        count = tyc_four_TwoPassLabel(thresSrc,tmp,result);
    }else{
        Mat tmp;
        count = tyc_eight_TwoPassLabel(thresSrc,tmp,result);
    }
    //保存图像
    m_saveMat.release();
    m_saveMat = result.clone();

    //显示连通域结果
    tyc_resize(result,LabelHeight3,LabelWigth3);
    cvtColor(result,result,COLOR_BGR2RGB);
    Result = QImage((const uchar*)(result.data), result.cols, result.rows,
                    result.cols*result.channels(),QImage::Format_RGB888);
    ui->label_output->setAlignment(Qt::AlignCenter);
    ui->label_output->setPixmap(QPixmap::fromImage(Result));
    ui->textEdit1->setText(QString::number(count));
    QMessageBox::warning(this, tr("image processing"), tr("finish processing!"));
    return;
}


//直方图读图
void Widget::on_pushButton_input1_2_clicked()
{
    using namespace cv;
    QString open_file, open_file_path;
    open_file = QFileDialog::getOpenFileName(this,
                                             "please choose an image file for hist",
                                             QDir::currentPath(),
                                             "Image Files(*.jpg *.png *.bmp);;All(*.*)");
    if(open_file.isNull()){
        QMessageBox::warning(this, tr("read image"), tr("failly read image!"));
        return;
    }
    qDebug()<<open_file<<endl;
    m_srcimgPath = open_file;
    Mat mat = imread(open_file.toStdString());
    tyc_resize(mat,LabelHeight3,LabelWigth3);
    cvtColor(mat,mat,COLOR_BGR2RGB);
    QImage Result;
    Result = QImage((const uchar*)(mat.data), mat.cols, mat.rows,
                    mat.cols*mat.channels(),QImage::Format_RGB888);
    ui->label_input->setAlignment(Qt::AlignCenter);
    ui->label_input->setPixmap(QPixmap::fromImage(Result));
    QMessageBox::warning(this, tr("read image"), tr("successfully read image!"));
}

void Widget::on_pushButton_deal1_2_clicked()
{
    if(!m_srcimgPath.size()){
        QMessageBox::warning(this, tr("histogram_equaliaztion"), tr("please input image!"));
        return;
    }
    // 灰度图像
    using namespace cv;
    Mat src;
    src = imread(m_srcimgPath.toStdString());
    m_srcimgPath.clear();
    cvtColor(src, src, COLOR_BGR2GRAY);
    Mat histSrc,histDeal,hist;
    histSrc = DrawHistImg(src);
    hist = src.clone();
    tyc_EquHistogram(src,hist);
    histDeal = DrawHistImg(hist);
    //保存图像
    m_saveMat.release();
    m_saveMat = hist.clone();
    //显示直方图
    tyc_resize(histSrc,LabelHeight2,LabelWigth2);
    cvtColor(histSrc,histSrc,COLOR_BGR2RGB);
    QImage Result;
    Result = QImage((const uchar*)(histSrc.data), histSrc.cols, histSrc.rows,
                    histSrc.cols*histSrc.channels(),QImage::Format_RGB888);
    ui->label_srcHisto->setAlignment(Qt::AlignCenter);
    ui->label_srcHisto->setPixmap(QPixmap::fromImage(Result));
    //显示结果直方图
    tyc_resize(histDeal,LabelHeight2,LabelWigth2);
    cvtColor(histDeal,histDeal,COLOR_BGR2RGB);
    Result = QImage((const uchar*)(histDeal.data), histDeal.cols, histDeal.rows,
                    histDeal.cols*histDeal.channels(),QImage::Format_RGB888);
    ui->label_resHisto->setAlignment(Qt::AlignCenter);
    ui->label_resHisto->setPixmap(QPixmap::fromImage(Result));
    //显示均衡化的图
    tyc_resize(hist,LabelHeight3,LabelWigth3);
    cvtColor(hist,hist,COLOR_GRAY2RGB);
    Result = QImage((const uchar*)(hist.data), hist.cols, hist.rows,
                    hist.cols*hist.channels(),QImage::Format_RGB888);
    ui->label_output->setAlignment(Qt::AlignCenter);
    ui->label_output->setPixmap(QPixmap::fromImage(Result));
    QMessageBox::warning(this, tr("image processing"), tr("finish processing!"));
}



void Widget::on_pushButton_input1_3_clicked()
{
    using namespace cv;
    QString open_file, open_file_path;
    open_file = QFileDialog::getOpenFileName(this,
                                             "please choose an image file for canny",
                                             QDir::currentPath(),
                                             "Image Files(*.jpg *.png *.bmp);;All(*.*)");
    if(open_file.isNull()){
        QMessageBox::warning(this, tr("read image"), tr("failly read image!"));
        return;
    }
    qDebug()<<open_file<<endl;
    m_srcimgPath = open_file;
    Mat mat = imread(open_file.toStdString());
    tyc_resize(mat,LabelHeight3,LabelWigth3);
    cvtColor(mat,mat,COLOR_BGR2RGB);
    QImage Result;
    Result = QImage((const uchar*)(mat.data), mat.cols, mat.rows,
                    mat.cols*mat.channels(),QImage::Format_RGB888);
    ui->label_input->setAlignment(Qt::AlignCenter);
    ui->label_input->setPixmap(QPixmap::fromImage(Result));
    QMessageBox::warning(this, tr("read image"), tr("successfully read image!"));
}

//canny处理
void Widget::on_pushButton_deal1_3_clicked()
{
    if(!m_srcimgPath.size()){
        QMessageBox::warning(this, tr("canny detection"), tr("please input image!"));
        return;
    }
    // 灰度图像
    using namespace cv;
    Mat src;
    src = imread(m_srcimgPath.toStdString());
    m_srcimgPath.clear();
    cvtColor(src, src, COLOR_BGR2GRAY);
    Mat definecanny;
    tyc_Canny(src,definecanny);
    //保存图像
    m_saveMat.release();
    m_saveMat = definecanny.clone();
    Mat edge;
    Canny(src, edge, 50, 150);
    //显示
    tyc_resize(definecanny,LabelHeight3,LabelWigth3);
    cvtColor(definecanny,definecanny,COLOR_GRAY2RGB);
    QImage Result;
    Result = QImage((const uchar*)(definecanny.data), definecanny.cols, definecanny.rows,
                    definecanny.cols*definecanny.channels(),QImage::Format_RGB888);
    ui->label_output->setAlignment(Qt::AlignCenter);
    ui->label_output->setPixmap(QPixmap::fromImage(Result));
    //显示opencv库的结果
    tyc_resize(edge,LabelHeight1,LabelWigth1);
    cvtColor(edge,edge,COLOR_GRAY2RGB);
    Result = QImage((const uchar*)(edge.data), edge.cols, edge.rows,
                    edge.cols*edge.channels(),QImage::Format_RGB888);
    ui->label_cv_canny->setAlignment(Qt::AlignCenter);
    ui->label_cv_canny->setPixmap(QPixmap::fromImage(Result));
    QMessageBox::warning(this, tr("image processing"), tr("finish processing!"));

}

void Widget::on_pushButton_input1_4_clicked()
{
    using namespace cv;
    QString open_file, open_file_path;
    open_file = QFileDialog::getOpenFileName(this,
                                             "please choose an image file for circle center det",
                                             QDir::currentPath(),
                                             "Image Files(*.jpg *.png *.bmp);;All(*.*)");
    if(open_file.isNull()){
        QMessageBox::warning(this, tr("read image"), tr("failly read image!"));
        return;
    }
    qDebug()<<open_file<<endl;
    m_srcimgPath = open_file;
    Mat mat = imread(open_file.toStdString());
    tyc_resize(mat,LabelHeight3,LabelWigth3);
    cvtColor(mat,mat,COLOR_BGR2RGB);
    QImage Result;
    Result = QImage((const uchar*)(mat.data), mat.cols, mat.rows,
                    mat.cols*mat.channels(),QImage::Format_RGB888);
    ui->label_input->setAlignment(Qt::AlignCenter);
    ui->label_input->setPixmap(QPixmap::fromImage(Result));
    QMessageBox::warning(this, tr("read image"), tr("successfully read image!"));
}

void Widget::on_pushButton_deal1_4_clicked()
{
    if(!m_srcimgPath.size()){
        QMessageBox::warning(this, tr("circle center detect"), tr("please input image!"));
        return;
    }
    using namespace cv;
    using namespace std;
    Mat src = imread(m_srcimgPath.toStdString());
    Mat res,tycback,components_;
    vector<std::pair<double,double>> detPoints;
    detPoints = tyc_det_ciclePos(src,res,tycback,components_);
    m_srcimgPath.clear();
    //保存图像
    m_saveMat.release();
    m_saveMat = res.clone();
    //显示结果图片
    tyc_resize(res,LabelHeight3,LabelWigth3);
    cvtColor(res,res,COLOR_BGR2RGB);
    QImage Result;
    Result = QImage((const uchar*)(res.data), res.cols, res.rows,
                    res.cols*res.channels(),QImage::Format_RGB888);
    ui->label_output->setAlignment(Qt::AlignCenter);
    ui->label_output->setPixmap(QPixmap::fromImage(Result));
    //显示中间过程
    tyc_resize(tycback,LabelHeight1,LabelWigth1);
    cvtColor(tycback,tycback,COLOR_GRAY2RGB);
    Result = QImage((const uchar*)(tycback.data), tycback.cols, tycback.rows,
                    tycback.cols*tycback.channels(),QImage::Format_RGB888);
    ui->label_background->setAlignment(Qt::AlignCenter);
    ui->label_background->setPixmap(QPixmap::fromImage(Result));
    //输出第二个
    tyc_resize(components_,LabelHeight1,LabelWigth1);
    cvtColor(components_,components_,COLOR_GRAY2RGB);
//    imshow("y",components_);
    Result = QImage((const uchar*)(components_.data), components_.cols, components_.rows,
                    components_.cols*components_.channels(),QImage::Format_RGB888);
    ui->label_Components->setAlignment(Qt::AlignCenter);
    ui->label_Components->setPixmap(QPixmap::fromImage(Result));
    ui->label_Components->show();
    QString tmp = "idx---colVal---rowVal";
    ui->textEdit1_4->append(tmp);
    for(size_t i = 0;i<detPoints.size();++i){
        tmp.clear();
        double col = detPoints[i].first;
        double row = detPoints[i].second;
        tmp = QString::number(i);
        tmp+="---";
        tmp+=QString::number(col);
        tmp+="---";
        tmp+=QString::number(row);
        ui->textEdit1_4->append(tmp);
    }
    qDebug()<<"finish"<<endl;
    QMessageBox::warning(this, tr("image processing"), tr("finish processing!"));
}



void Widget::on_pushButton_clicked()
{
    //首先生成随即图
    int sizeCount = ui->lineEdit->text().toUInt();
    qDebug()<<sizeCount<<endl;
    assert(sizeCount>0);
    using namespace cv;
    Mat img(sizeCount, sizeCount, CV_8UC1);
    for (int i = 0; i < sizeCount; ++i) {
        for (int j = 0; j < sizeCount; ++j) {
            int tmp = ((rand() % 100) > 50 ? 0 : 255);
            img.at<uchar>(i, j) = tmp;
        }
    }
    int idx = ui->comboBox->currentIndex();
    Mat result;
    int count = 0;
    int opencvCpunt = 0;
    if(idx){
        Mat tmp,out;
        count = tyc_four_TwoPassLabel(img,tmp,result);
        opencvCpunt = connectedComponents(img, out, 4, CV_32S);
    }else{
        Mat tmp,out;;
        count = tyc_eight_TwoPassLabel(img,tmp,result);
        opencvCpunt = connectedComponents(img, out, 8, CV_32S);
    }
    //保存图像
    m_saveMat.release();
    m_saveMat = result.clone();

    //显示结果
    ui->textEdit_mycount->setText(QString::number(count));
    ui->textEdit_cvcount->setText(QString::number(opencvCpunt));
    //显示结果
    tyc_resize(result,LabelHeight3,LabelWigth3);
    cvtColor(result,result,COLOR_BGR2RGB);
    QImage Result;
    Result = QImage((const uchar*)(result.data), result.cols, result.rows,
                    result.cols*result.channels(),QImage::Format_RGB888);
    ui->label_output->setAlignment(Qt::AlignCenter);
    ui->label_output->setPixmap(QPixmap::fromImage(Result));
    //显示原图
    tyc_resize(img,LabelHeight3,LabelWigth3);
    cvtColor(img,img,COLOR_GRAY2RGB);
    Result = QImage((const uchar*)(img.data), img.cols, img.rows,
                    img.cols*img.channels(),QImage::Format_RGB888);
    ui->label_input->setAlignment(Qt::AlignCenter);
    ui->label_input->setPixmap(QPixmap::fromImage(Result));
    QMessageBox::warning(this, tr("image processing"), tr("finish processing!"));
}

void Widget::on_pushButton_saveimage_clicked()
{
    if(!m_saveDir.size()){
        QString fileName = QCoreApplication::applicationDirPath();
        QDateTime time = QDateTime::currentDateTime();//获取系统现在的时间
        QString str = time.toString("yyyy_MM_dd_hh_mm_ss"); //设置显示格式
        QDir *folder = new QDir;
        QString newdir=fileName+"/"+str;
        qDebug()<<newdir<<endl;
        folder->mkdir(newdir);
        m_saveDir = newdir;
    }
    if(m_saveMat.rows){
        using namespace cv;
        QDateTime time = QDateTime::currentDateTime();//获取系统现在的时间
        QString str = time.toString("hh_mm_ss"); //设置显示格式
        QString TMP = m_saveDir+"/"+str+".bmp";
        imwrite(TMP.toStdString(),m_saveMat);
        m_saveMat.release();
        QMessageBox::warning(this, tr("image saving"), tr("finish saving!"));
    }else{
        qDebug()<<"input image"<<endl;
    }

}









