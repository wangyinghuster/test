#include "videoInput.h"
#include "opencv2/imgproc/imgproc.hpp"

videoInput::videoInput()
    :videoName("vedioBase")
{
}

videoInput::~videoInput(){} //�����������������ṩһ������

string videoInput::getVideoName()
{
    return videoName;
}

//virtual ����û�б�Ҫ��������������������Ҳ����������
void videoInput::update(Mat &frameIn)
{
    frame=frameIn;
}

void videoInput::getBackground(Mat &BGOut)
{
    BGOut=backGround;
}

void videoInput::getForeMask(Mat &FMOut)
{
    FMOut=foreMask;
}

void videoInput::getFrame(Mat &frameOut)
{
    frameOut=frame;
}

/*************************************videoMOGInput****************************************/
videoMOGInput::videoMOGInput()
{
	//���ͺ˺���

	Scalar color(255,255,255);
	int a=3;             //����˵�ֵӦ�ú�ȫͼ�Ĵ�С�ɱ�����Ӧ,seamRadio�Ǳ���
	Mat paint=Mat::zeros(2*a+1,2*a+1,CV_8UC3);
	Point cnt1(a+1,a+1);
	circle(paint,cnt1,a,color,CV_FILLED);   //��������ģ��
	cvtColor(paint,edCore,CV_RGB2GRAY);  //���ʹ���˵�����

	//��ʴ�˺���
	diCore=edCore; //��ʴ����˵�����

}

videoMOGInput::~videoMOGInput(){} //�����������������ṩһ������

void videoMOGInput::updateBackground()
{
    mog(frame,foreMask);
}

void videoMOGInput::getBackground(Mat &BGOut)
{
    mog.getBackgroundImage(backGround);
	videoInput::getBackground(BGOut);
}

void videoMOGInput::getForeMask(Mat &FMOut)
{
	mog(frame,FMOut);
	erode(FMOut,foreMask,edCore);  //��ʴ
	dilate(foreMask,FMOut,diCore,Point(-1,-1),3); // ����
}

/**************************************videoAVGInput***************************************/
videoAVGInput::videoAVGInput()
    :count(0)
{

}

videoAVGInput::~videoAVGInput(){}

void videoAVGInput::updateBackground(){
    if (imgAVG.empty()) //�ǵó�ʼ��
		imgAVG=Mat::zeros(frame.rows,frame.cols,CV_32FC3); //imgAVG.create(frame.rows,frame.cols,CV_32FC3);

    if (frame.type()!=CV_8UC3) throw sysException("Input frame mast be CV_8UC3!");
    Mat temp;
    frame.convertTo(temp,CV_32FC3); //����Ҫ��frame������CV_8UC3��ʽ��
    imgAVG+=temp;
    ++count;
}

void videoAVGInput::getBackground(Mat &BGOut){
    if (!count) throw sysException("You need to push frame before getting background!");

    imgAVG=imgAVG/count;
    imgAVG.convertTo(BGOut,CV_8UC3);

}

void videoAVGInput::getForeMask(Mat &FMOut)  
{
	throw sysException("���ﻹû��д�ã�������Ҫ�ã���");
	Mat temp;
	frame.convertTo(temp,CV_32FC3); //����Ҫ��frame������CV_8UC3��ʽ��
	temp=frame-imgAVG;
	temp.convertTo(FMOut,CV_8UC3);
	erode(FMOut,foreMask,Mat());  //��ʴ
	dilate(foreMask,FMOut,Mat()); // ����
}









