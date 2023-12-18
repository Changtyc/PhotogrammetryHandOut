#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include<QString>
#include <opencv2/opencv.hpp>
#include <ImgAlgo.h>


QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();

private slots:
    void on_pushButton_input1_clicked();

    void on_pushButton_deal1_clicked();

    void on_pushButton_input1_2_clicked();

    void on_pushButton_deal1_2_clicked();

    void on_pushButton_input1_3_clicked();

    void on_pushButton_deal1_3_clicked();

    void on_pushButton_input1_4_clicked();

    void on_pushButton_deal1_4_clicked();

    void on_pushButton_clicked();

    void on_pushButton_saveimage_clicked();

private:
    Ui::Widget *ui;


private:
    QString m_srcimgPath;
    cv::Mat m_saveMat;
    QString m_saveDir="";

public:
    const int LabelHeight1 = 600;
    const int LabelWigth1 = 600;
    const int LabelHeight2 = 400;
    const int LabelWigth2 = 480;
    const int LabelHeight3 = 750;
    const int LabelWigth3 = 640;

};
#endif // WIDGET_H
