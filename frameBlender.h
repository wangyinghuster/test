#ifndef FRAMEBLENDER_H
#define FRAMEBLENDER_H

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

#include "macrosConfig.h" //预编译宏配置文件

using namespace std;
using namespace cv;

class frameBlender
{
public:
    frameBlender();
    virtual ~frameBlender();

	virtual void prepare(vector<Mat> &warpSeamMask,vector<Mat> &warpMask,vector<Point> &topleft)=0; //     用于计算准备参数  输入：warped seam masks，warped masks，topleft顶点
	virtual void doBlend(vector<Mat> &warpImg,vector<Point> &topleft,Mat &panoImgOut)=0;//     用于用参数计算blend输入：warped images输出：warped images，全景图panorama
	virtual void adjustForground(vector<Mat> &warpImg,vector<Mat> warpForeMsk,vector<Point> topleft)=0;; // 以后定义    用于调整 输入：前景warp mask ，warped images 输出：全景图panorama

    bool isPerpared(){return prepared;}  //在这里定义的简单函数自动内联

protected:
    Point getAllSize(vector<Mat> &warpSeamMask,vector<Point> &topleft); //这里面会顺带初始化outCols和outRows
    bool prepared;

    int outRows;            //最终图像的行数
    int outCols;            //最终图像的列数

};

#endif // FRAMEBLENDER_H
