#ifndef FEATURESGENERATOR_H
#define FEATURESGENERATOR_H

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include "macrosConfig.h" //预编译宏配置文件

using namespace std;
using namespace cv;

class imgFeatures
{
public:
    Mat backGroundFeature;   //背景特征
    vector<KeyPoint> backGroundPoint;   //背景特征点
    string method;           //指定特征点提取方法
};

class featuresGenerator
{
public:
    featuresGenerator();
    virtual ~featuresGenerator()=0;

public:
    virtual void detectFeature(Mat &image,imgFeatures &feature);

protected:
    string method;
    
	static int colsMax;   //特征分块计算输入的图像的列数最大值
	static int rowsMax;    //特征分块计算输入图像的行数最大值
};

class SIFTfeaturesGenerator:public featuresGenerator
{
public:
    SIFTfeaturesGenerator(){method="SIFT"; initModule_nonfree(); }
};

class SURFfeaturesGenerator:public featuresGenerator
{
public:
    SURFfeaturesGenerator(){method="SURF"; initModule_nonfree(); }
};

class ORBfeaturesGenerator:public featuresGenerator
{
public:
    ORBfeaturesGenerator(){method="ORB";}
    virtual void detectFeature(Mat &image,imgFeatures &feature);

};


#endif // FEATURESGENERATOR_H
