#ifndef FEATURESGENERATOR_H
#define FEATURESGENERATOR_H

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include "macrosConfig.h" //Ԥ����������ļ�

using namespace std;
using namespace cv;

class imgFeatures
{
public:
    Mat backGroundFeature;   //��������
    vector<KeyPoint> backGroundPoint;   //����������
    string method;           //ָ����������ȡ����
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
    
	static int colsMax;   //�����ֿ���������ͼ����������ֵ
	static int rowsMax;    //�����ֿ��������ͼ����������ֵ
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
