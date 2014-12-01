#ifndef DIRECTBLEND_H
#define DIRECTBLEND_H

#include "frameBlender.h"

class directBlender : public frameBlender//直接对重叠区域进行融合不找缝
{
public:
    //directBlender();

public:
    void prepare(vector<Mat> &warpSeamMask, vector<Mat> &warpMask, vector<Point> &topleft);
    void doBlend(vector<Mat> &warpImg,vector<Point> &topleft,Mat &panoImgOut);
	void adjustForground(vector<Mat> &warpImg,vector<Mat> warpForeMsk,vector<Point> topleft);
private:
	vector<Mat> imgMaskWeight;  //图像权重
	vector<Mat> seamMasks_; //保存warpSeamMask供adjustForground使用
	vector<Mat> warpForeMasks_;//保存warpForeMasks供doBlend使用         要改doBlend接口??????
	vector<Mat> foreMasks_;  //前景图像蒙版
	Mat foreOut_;
	Mat grayImgOut_;
	vector<Mat> grayImgs_;
};
#endif