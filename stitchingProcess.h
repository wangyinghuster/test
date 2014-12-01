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

	void inputVideo(int idx,Mat &img){(this->*inputPtr)(idx,img);}  //这个函数准备用函数指针来实现

private:
    void inputVideoBeforePrepare(int idx,Mat &img);
    void inputVideoAfterPrepare(int idx,Mat &img);
    
    void (stitchingProcess::*inputPtr)(int idx,Mat &img);  //用于实现的函数指针

private:
    void updateIdx(vector<bool> &flags);

private:

    int videoNum;
    videoInput **videos;   //视屏拼接指针
    featuresGenerator *fGnerator;  //特征生成描述子
    featuresMatcher *fMatch;   //特征匹配描述子
    frameWarper *fWarper;
    seamFinder *sFinder;      //拼接缝寻找
    frameBlender *fBlender;    //融合

    Mat idxMap;        //存放视屏是否存在于匹配集合中,以及索引的映射值
	char *vIdx; //数据指针，指向idxMap的内部数据，时刻记住这个在idxMap修改之后必须要重新赋值
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
