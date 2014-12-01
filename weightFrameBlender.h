#ifndef WEIGHTBLEND_H
#define WEIGHTBLEND_H

#include "frameBlender.h"

class weightFrameBlender : public frameBlender
{
public:
    weightFrameBlender(int lNum = 5,int sRadio = 20);

public:

    void prepare(vector<Mat> &warpSeamMask, vector<Mat> &warpMask, vector<Point> &topleft);
    void doBlend(vector<Mat> &warpImg,vector<Point> &topleft,Mat &panoImgOut);
	void adjustForground(vector<Mat> &warpImg,vector<Mat> warpForeMsk,vector<Point> topleft);

private:
    vector<Mat> imgMaskWeight;  //图像权重
	
    int layerNum;
    int seamRadio;            //融合痕迹的宽度

private:
	vector<Mat> overMask; //保存前景蒙版
    Mat foregroudMap;  //前景图像蒙版
	static const int foreUse=254;  //两个值，对应前景蒙版中使用或者是不使用
	static const int foreNotUse=128;
	static float destTher;  //前景中心的相差阈值

};

#endif // WEIGHTBELND_H
