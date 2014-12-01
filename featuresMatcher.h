#ifndef FEATURESMATCHER_H
#define FEATURESMATCHER_H

#include <iostream>
#include <string>
#include <vector>

#include "featuresGenerator.h"
#include "macrosConfig.h" //预编译宏配置文件

using namespace std;
using namespace cv;

typedef struct matchLog{    //传递测参数类
    int queryInx;   //查询的图像号
    int trainInx;   //被查询的图像号
    vector<DMatch> matchPointIndex;  //特征点集合
    vector<DMatch> matchPointIndexRev;  //特征点集合，query和train交换的

  //用匹配点的数目作为权值，进行排序
    bool operator < (const struct matchLog &m)const {
        return matchPointIndex.size() < m.matchPointIndex.size();
    }  //为了排序而重载的操作符

} mLog;

class featuresMatcher   //基类
{
public:
    featuresMatcher();
	virtual ~featuresMatcher()=0;

public:
    virtual void buildMatch(vector<Mat> &imgSet,vector<imgFeatures> &fSet,vector<mLog> &matchInfoOut);
    virtual void findSeperatedMatchSets(Mat &setsFlagOut);
    virtual void findLargestSets(vector<Mat> &imgSetInOut,vector<imgFeatures> &fSetInOut,vector<mLog> &matchInfoInOut,vector<bool> &videoIdxOut);

protected:
    float matchThr;
    Mat matchMap;      //采用匹配表的形式保存匹配的关系表
    Mat setsFlag;         //集合代号
    Mat idxMap;           //映射索引的值
};

class SIFTfeaturesMatcher:public featuresMatcher
{
public:
    SIFTfeaturesMatcher(){ matchThr=0.6;}
};

class SURFfeaturesMatcher:public featuresMatcher
{
public:
    SURFfeaturesMatcher(){ matchThr=0.35;}
};

class ORBfeaturesMatcher:public featuresMatcher
{
public:
    ORBfeaturesMatcher(){ matchThr=0.7;}
};

#endif // FEATURESMATCHER_H
