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
    vector<Mat> imgMaskWeight;  //ͼ��Ȩ��
	
    int layerNum;
    int seamRadio;            //�ںϺۼ��Ŀ��

private:
	vector<Mat> overMask; //����ǰ���ɰ�
    Mat foregroudMap;  //ǰ��ͼ���ɰ�
	static const int foreUse=254;  //����ֵ����Ӧǰ���ɰ���ʹ�û����ǲ�ʹ��
	static const int foreNotUse=128;
	static float destTher;  //ǰ�����ĵ������ֵ

};

#endif // WEIGHTBELND_H
