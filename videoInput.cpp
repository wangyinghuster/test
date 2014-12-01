#include "videoInput.h"
#include "opencv2/imgproc/imgproc.hpp"

videoInput::videoInput()
    :videoName("vedioBase")
{
}

videoInput::~videoInput(){} //纯虚析构函数必须提供一个定义

string videoInput::getVideoName()
{
    return videoName;
}

//virtual 函数没有必要声明成内联。。声明了也不会内联的
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
	//膨胀核函数

	Scalar color(255,255,255);
	int a=3;             //这个核的值应该和全图的大小成比例对应,seamRadio是比例
	Mat paint=Mat::zeros(2*a+1,2*a+1,CV_8UC3);
	Point cnt1(a+1,a+1);
	circle(paint,cnt1,a,color,CV_FILLED);   //生成球形模板
	cvtColor(paint,edCore,CV_RGB2GRAY);  //膨胀处理核的生成

	//腐蚀核函数
	diCore=edCore; //腐蚀处理核的生成

}

videoMOGInput::~videoMOGInput(){} //纯虚析构函数必须提供一个定义

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
	erode(FMOut,foreMask,edCore);  //腐蚀
	dilate(foreMask,FMOut,diCore,Point(-1,-1),3); // 膨胀
}

/**************************************videoAVGInput***************************************/
videoAVGInput::videoAVGInput()
    :count(0)
{

}

videoAVGInput::~videoAVGInput(){}

void videoAVGInput::updateBackground(){
    if (imgAVG.empty()) //记得初始化
		imgAVG=Mat::zeros(frame.rows,frame.cols,CV_32FC3); //imgAVG.create(frame.rows,frame.cols,CV_32FC3);

    if (frame.type()!=CV_8UC3) throw sysException("Input frame mast be CV_8UC3!");
    Mat temp;
    frame.convertTo(temp,CV_32FC3); //这里要求frame必须是CV_8UC3格式的
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
	throw sysException("这里还没有写好！！！不要用！！");
	Mat temp;
	frame.convertTo(temp,CV_32FC3); //这里要求frame必须是CV_8UC3格式的
	temp=frame-imgAVG;
	temp.convertTo(FMOut,CV_8UC3);
	erode(FMOut,foreMask,Mat());  //腐蚀
	dilate(foreMask,FMOut,Mat()); // 膨胀
}









