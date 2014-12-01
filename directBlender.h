#ifndef DIRECTBLEND_H
#define DIRECTBLEND_H

#include "frameBlender.h"

class directBlender : public frameBlender//ֱ�Ӷ��ص���������ںϲ��ҷ�
{
public:
    //directBlender();

public:
    void prepare(vector<Mat> &warpSeamMask, vector<Mat> &warpMask, vector<Point> &topleft);
    void doBlend(vector<Mat> &warpImg,vector<Point> &topleft,Mat &panoImgOut);
	void adjustForground(vector<Mat> &warpImg,vector<Mat> warpForeMsk,vector<Point> topleft);
private:
	vector<Mat> imgMaskWeight;  //ͼ��Ȩ��
	vector<Mat> seamMasks_; //����warpSeamMask��adjustForgroundʹ��
	vector<Mat> warpForeMasks_;//����warpForeMasks��doBlendʹ��         Ҫ��doBlend�ӿ�??????
	vector<Mat> foreMasks_;  //ǰ��ͼ���ɰ�
	Mat foreOut_;
	Mat grayImgOut_;
	vector<Mat> grayImgs_;
};
#endif