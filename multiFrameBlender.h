#ifndef MULTIFRAMEBLENDER_H
#define MULTIFRAMEBLENDER_H

#include "frameBlender.h"

using namespace std;
using namespace cv;

class multiFrameBlender : public frameBlender
{
public:
    multiFrameBlender(int num_bands=5, bool tryGpu=false);

    virtual void prepare(vector<Mat> &warpSeamMask,vector<Mat> &warpMask,vector<Point> &topleft);//     ���ڼ���׼������  ���룺warped seam masks��warped masks��topleft����
    virtual void doBlend(vector<Mat> &warpImg,vector<Point> &topleft,Mat &panoImgOut);//     �����ò�������blend���룺warped images�����warped images��ȫ��ͼpanorama
	virtual void adjustForground(){}//     ���ڵ��� ���룺ǰ��warp mask ��warped images �����ȫ��ͼpanorama

private:
    int numBands() const { return actual_num_bands_; }
    void setNumBands(int val) { actual_num_bands_ = val; }

    void prepare(Rect dst_roi);
    void feed(const Mat &img);

    void blend(Mat &dst, Mat &dst_mask);  //���ص�������CV_16SC3

    void clearImg();

private:
    void bulitMaskPyr(const Mat &mask,Point tl);  //������˹Mask������
    void normalizeWeightPyr();  //�Ը�˹MASK���������й�һ��

struct borderType{
    Point2i tl;
    Point2i br;
    int top;
    int left;
    int bottom;
    int right;
};
    vector<borderType> borders;  //���������

    Mat dst_, dst_mask_;
    Rect dst_roi_;
    int actual_num_bands_, num_bands_;
    vector<Mat> dst_pyr_laplace_;       //ȫ��ͼ���MASK������
    vector<Mat> dst_band_weights_;  //ȫ�ֵ�ͼ��MASK����������
    vector<vector<Mat> > imgs_mask_pyr_gauss; //�����һ�������ͼ�����е�MASK������
    Rect dst_roi_final_;
    int feed_count_;
   // int weight_type_; //CV_32F or CV_16S
    bool useGpu;

};


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

void normalizeUsingWeightMap(const Mat& weight, Mat& src);

void createLaplacePyr(const Mat &img, int num_levels, std::vector<Mat>& pyr);

void createLaplacePyrGpu(const Mat &img, int num_levels, std::vector<Mat> &pyr);

// Restores source image
void restoreImageFromLaplacePyr(std::vector<Mat>& pyr);

void myPyrUp(const Mat &input, Mat &output);

void myLaplacePyr(const Mat &downInput, const Mat &source, Mat &outImg);

void myPyrDown(const Mat &input, Mat &output);

//void mySubstrate(Mat &input1,Mat &input2,Mat output);

#endif // MULTIFRAMEBLENDER_H
