#ifndef FEATURESMATCHER_H
#define FEATURESMATCHER_H

#include <iostream>
#include <string>
#include <vector>

#include "featuresGenerator.h"
#include "macrosConfig.h" //Ԥ����������ļ�

using namespace std;
using namespace cv;

typedef struct matchLog{    //���ݲ������
    int queryInx;   //��ѯ��ͼ���
    int trainInx;   //����ѯ��ͼ���
    vector<DMatch> matchPointIndex;  //�����㼯��
    vector<DMatch> matchPointIndexRev;  //�����㼯�ϣ�query��train������

  //��ƥ������Ŀ��ΪȨֵ����������
    bool operator < (const struct matchLog &m)const {
        return matchPointIndex.size() < m.matchPointIndex.size();
    }  //Ϊ����������صĲ�����

} mLog;

class featuresMatcher   //����
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
    Mat matchMap;      //����ƥ������ʽ����ƥ��Ĺ�ϵ��
    Mat setsFlag;         //���ϴ���
    Mat idxMap;           //ӳ��������ֵ
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
