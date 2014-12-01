#ifndef FRAMEBLENDER_H
#define FRAMEBLENDER_H

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

#include "macrosConfig.h" //Ԥ����������ļ�

using namespace std;
using namespace cv;

class frameBlender
{
public:
    frameBlender();
    virtual ~frameBlender();

	virtual void prepare(vector<Mat> &warpSeamMask,vector<Mat> &warpMask,vector<Point> &topleft)=0; //     ���ڼ���׼������  ���룺warped seam masks��warped masks��topleft����
	virtual void doBlend(vector<Mat> &warpImg,vector<Point> &topleft,Mat &panoImgOut)=0;//     �����ò�������blend���룺warped images�����warped images��ȫ��ͼpanorama
	virtual void adjustForground(vector<Mat> &warpImg,vector<Mat> warpForeMsk,vector<Point> topleft)=0;; // �Ժ���    ���ڵ��� ���룺ǰ��warp mask ��warped images �����ȫ��ͼpanorama

    bool isPerpared(){return prepared;}  //�����ﶨ��ļ򵥺����Զ�����

protected:
    Point getAllSize(vector<Mat> &warpSeamMask,vector<Point> &topleft); //�������˳����ʼ��outCols��outRows
    bool prepared;

    int outRows;            //����ͼ�������
    int outCols;            //����ͼ�������

};

#endif // FRAMEBLENDER_H
