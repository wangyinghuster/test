#ifndef MULTIFRAMEBLENDER_H
#define MULTIFRAMEBLENDER_H

#include "frameBlender.h"

using namespace std;
using namespace cv;

class multiFrameBlender : public frameBlender
{
public:
    multiFrameBlender(int num_bands=5, bool tryGpu=false);

    virtual void prepare(vector<Mat> &warpSeamMask,vector<Mat> &warpMask,vector<Point> &topleft);//     用于计算准备参数  输入：warped seam masks，warped masks，topleft顶点
    virtual void doBlend(vector<Mat> &warpImg,vector<Point> &topleft,Mat &panoImgOut);//     用于用参数计算blend输入：warped images输出：warped images，全景图panorama
	virtual void adjustForground(){}//     用于调整 输入：前景warp mask ，warped images 输出：全景图panorama

private:
    int numBands() const { return actual_num_bands_; }
    void setNumBands(int val) { actual_num_bands_ = val; }

    void prepare(Rect dst_roi);
    void feed(const Mat &img);

    void blend(Mat &dst, Mat &dst_mask);  //返回的类型是CV_16SC3

    void clearImg();

private:
    void bulitMaskPyr(const Mat &mask,Point tl);  //创建高斯Mask金字塔
    void normalizeWeightPyr();  //对高斯MASK金字塔进行归一化

struct borderType{
    Point2i tl;
    Point2i br;
    int top;
    int left;
    int bottom;
    int right;
};
    vector<borderType> borders;  //保存计算结果

    Mat dst_, dst_mask_;
    Rect dst_roi_;
    int actual_num_bands_, num_bands_;
    vector<Mat> dst_pyr_laplace_;       //全局图像的MASK金字塔
    vector<Mat> dst_band_weights_;  //全局的图像MASK金字塔保存
    vector<vector<Mat> > imgs_mask_pyr_gauss; //保存归一化过后的图像序列的MASK金字塔
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
