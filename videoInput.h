#ifndef VIDEOINPUT_H
#define VIDEOINPUT_H

/**************************************************
  该类创建一个视频序列对象，支持从device或者从video文件输入。
  功能：
  1.提取背景图像（GMM）
  2.计算背景图像的特征（ORB,SIFT,SURF）
  3.提取前景图像，运动检测
***************************************************/
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/background_segm.hpp>

#include "macrosConfig.h" //预编译宏配置文件

using namespace std;
using namespace cv;

class videoInput    //这个是基类
{
public:
    videoInput();
    virtual ~videoInput()=0;

public:
    virtual void update(Mat &frameIn);
    virtual void updateBackground()=0;
    virtual void getFrame(Mat &frameOut);
    virtual void getBackground(Mat &BGOut);
    virtual void getForeMask(Mat &FMOut);

    string getVideoName();

protected:
    Mat frame;
    Mat backGround;
    Mat foreMask;
    string videoName;

};

class videoMOGInput : public videoInput   //采用混合高斯背景模型的类
{
public:
    videoMOGInput();
    virtual ~videoMOGInput()=0;

    virtual void updateBackground();
	virtual void getBackground(Mat &BGOut);
	virtual void getForeMask(Mat &FMOut);

private:
    BackgroundSubtractorMOG2 mog;//背景建模对象
	Mat edCore; //腐蚀核
	Mat diCore; //膨胀核

};

class videoAVGInput : public videoInput   //均值背景建模
{
public:
    videoAVGInput();
    virtual ~videoAVGInput()=0;

    virtual void updateBackground();
    virtual void getBackground(Mat &BGOut);
	virtual void getForeMask(Mat &FMOut);

private:
    Mat imgAVG;
    int count;
};

#endif // VIDEOINPUT_H
