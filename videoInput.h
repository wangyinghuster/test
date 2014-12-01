#ifndef VIDEOINPUT_H
#define VIDEOINPUT_H

/**************************************************
  ���ഴ��һ����Ƶ���ж���֧�ִ�device���ߴ�video�ļ����롣
  ���ܣ�
  1.��ȡ����ͼ��GMM��
  2.���㱳��ͼ���������ORB,SIFT,SURF��
  3.��ȡǰ��ͼ���˶����
***************************************************/
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/background_segm.hpp>

#include "macrosConfig.h" //Ԥ����������ļ�

using namespace std;
using namespace cv;

class videoInput    //����ǻ���
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

class videoMOGInput : public videoInput   //���û�ϸ�˹����ģ�͵���
{
public:
    videoMOGInput();
    virtual ~videoMOGInput()=0;

    virtual void updateBackground();
	virtual void getBackground(Mat &BGOut);
	virtual void getForeMask(Mat &FMOut);

private:
    BackgroundSubtractorMOG2 mog;//������ģ����
	Mat edCore; //��ʴ��
	Mat diCore; //���ͺ�

};

class videoAVGInput : public videoInput   //��ֵ������ģ
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
