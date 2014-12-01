#ifndef STITCHINGPROCESS_H
#define STITCHINGPROCESS_H

#include <iostream>
#include <string>
#include <vector>

#include "videoInput.h"
#include "deviceVideoInput.h"
#include "aviVideoInput.h"
#include "featuresGenerator.h"
#include "featuresMatcher.h"
#include "frameWarper.h"
#include "parallaxWarper.h"
#include "seamFinder.h"
#include "opCVseamFinder.h"
#include "myCVseamFinder.h"
#include "frameBlender.h"
#include "multiFrameBlender.h"
#include "weightFrameBlender.h"
#include "directBlender.h"
#include "overlapSeamFind.h"

#include "macrosConfig.h"

using namespace std;
using namespace cv;

class stitchingProcess
{
public:
    stitchingProcess();
    ~stitchingProcess();

public:
    void generateVideoInput(int num);

    void stitch(Mat &imgOut);
    void prepare(Mat &imgEmptyOut);

	void inputVideo(int idx,Mat &img){(this->*inputPtr)(idx,img);}  //�������׼���ú���ָ����ʵ��

private:
    void inputVideoBeforePrepare(int idx,Mat &img);
    void inputVideoAfterPrepare(int idx,Mat &img);
    
    void (stitchingProcess::*inputPtr)(int idx,Mat &img);  //����ʵ�ֵĺ���ָ��

private:
    void updateIdx(vector<bool> &flags);

private:

    int videoNum;
    videoInput **videos;   //����ƴ��ָ��
    featuresGenerator *fGnerator;  //��������������
    featuresMatcher *fMatch;   //����ƥ��������
    frameWarper *fWarper;
    seamFinder *sFinder;      //ƴ�ӷ�Ѱ��
    frameBlender *fBlender;    //�ں�

    Mat idxMap;        //��������Ƿ������ƥ�伯����,�Լ�������ӳ��ֵ
	char *vIdx; //����ָ�룬ָ��idxMap���ڲ����ݣ�ʱ�̼�ס�����idxMap�޸�֮�����Ҫ���¸�ֵ
    vector<Mat>  imgSet;
    vector<mLog> imgMatchInfo;
    vector<imgFeatures> imgFeature;

    vector<Mat>  imgWarp;
    vector<Mat>  imgMaskWarp;
    vector<Mat>  imgSeamMask;
    vector<Point>  topleft;

    bool isPrepared;

    int outCols;
    int outRows;




};

#endif // STITCHINGPROCESS_H
