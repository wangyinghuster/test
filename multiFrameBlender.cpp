#include "multiFrameBlender.h"
#include "opencv2/imgproc/imgproc.hpp"


#ifndef __NOT_USE_OCL_LIB
#include "opencv2/ocl/ocl.hpp"
#endif

#ifndef __NOT_USE_OCL_LIB
using namespace ocl;
#endif

multiFrameBlender::multiFrameBlender(int num_bands,bool tryGpu)  //计划中使用OpenCL加速
    :useGpu(false)
{
    setNumBands(num_bands);

#ifndef __NOT_USE_OCL_LIB
    if (tryGpu){
        useGpu=true;
        vector<Info> oclInfo;
        int numDev=getDevice(oclInfo);
        if (numDev==0) {
            useGpu=false;
        }else{
        //if you want to use undefault device, set it here
        setDevice(oclInfo[0]);
        //set this to save kernel compile time from second time you run
        //setBinpath("./");

        __debug(cout << "[Info]Use Gpu to calculate." << endl;)

        }
    }
#endif
}


void multiFrameBlender::prepare(vector<Mat> &warpSeamMask, vector<Mat> &warpMask, vector<Point> &topleft)
{
    getAllSize(warpSeamMask,topleft);
    prepare(Rect(0,0,outCols,outRows));
    for (unsigned int k=0;k<warpSeamMask.size();++k){
        bulitMaskPyr(warpSeamMask[k],topleft[k]);
    }

    for (int i = 0; i <= num_bands_; ++i)
        normalizeUsingWeightMap(dst_band_weights_[i], dst_pyr_laplace_[i]);  //这里第一次做，就要归一化一下
    normalizeWeightPyr();  //归一化MASK的值

    prepared=true;
}


void multiFrameBlender::doBlend(vector<Mat> &warpImg,vector<Point> &topleft,Mat &panoImgOut)
{

    if (!prepared) throw sysException("Need to initialize befor blend!");
    clearImg();
    for (unsigned int k=0;k<warpImg.size();k++){
        feed(warpImg[k]);
    }

    Mat outImg;
    blend(outImg,Mat());
    outImg.convertTo(panoImgOut,CV_8U); //这里需要把图像深度给转换一下
}

//#define __DEBUG_DETAIL_INFO
static const float WEIGHT_EPS = 1e-5f;  //防止0溢出么？

void multiFrameBlender::prepare(Rect dst_roi)
{
    dst_roi_final_ = dst_roi;

    // Crop unnecessary bands
    double max_len = static_cast<double>(max(dst_roi.width, dst_roi.height));
    num_bands_ = min(actual_num_bands_, static_cast<int>(ceil(log(max_len) / log(2.0))));

    // Add border to the final image, to ensure sizes are divided by (1 << num_bands_)
    dst_roi.width += ((1 << num_bands_) - dst_roi.width % (1 << num_bands_)) % (1 << num_bands_);
    dst_roi.height += ((1 << num_bands_) - dst_roi.height % (1 << num_bands_)) % (1 << num_bands_);

    dst_.create(dst_roi.size(), CV_16SC3);
    dst_.setTo(Scalar::all(0));
    dst_mask_.create(dst_roi.size(), CV_8U);
    dst_mask_.setTo(Scalar::all(0));
    dst_roi_ = dst_roi;

    dst_pyr_laplace_.resize(num_bands_ + 1);
    dst_pyr_laplace_[0] = dst_;

    dst_band_weights_.resize(num_bands_ + 1);
    //dst_band_weights_[0].create(dst_roi.size(), weight_type_);
    dst_band_weights_[0].create(dst_roi.size(), CV_32F);
    dst_band_weights_[0].setTo(0);

    for (int i = 1; i <= num_bands_; ++i)
    {
        dst_pyr_laplace_[i].create((dst_pyr_laplace_[i - 1].rows + 1) / 2,
                                   (dst_pyr_laplace_[i - 1].cols + 1) / 2, CV_16SC3);
        dst_band_weights_[i].create((dst_band_weights_[i - 1].rows + 1) / 2,
                                     (dst_band_weights_[i - 1].cols + 1) / 2, CV_32F);
        dst_pyr_laplace_[i].setTo(Scalar::all(0));
        dst_band_weights_[i].setTo(0);
    }
    feed_count_=0;
}

void multiFrameBlender::blend(Mat &dst, Mat &dst_mask)
{

    restoreImageFromLaplacePyr(dst_pyr_laplace_);

    dst_ = dst_pyr_laplace_[0];
    dst_ = dst_(Range(0, dst_roi_final_.height), Range(0, dst_roi_final_.width));
    dst_mask_ = dst_band_weights_[0] > WEIGHT_EPS;
    dst_mask_ = dst_mask_(Range(0, dst_roi_final_.height), Range(0, dst_roi_final_.width));

    dst_.setTo(Scalar::all(0), dst_mask_ == 0);
    dst = dst_;
    dst_mask = dst_mask_;
}

void multiFrameBlender::clearImg(){

#ifdef __DEBUG_DETAIL_INFO
    clock_t c_start,c_end;
    c_start=clock();
#endif

    feed_count_=0;
    for (int i = 0; i <= num_bands_; ++i){  //仅仅清除结果图像
        dst_pyr_laplace_[i].setTo(Scalar::all(0));
    }

#ifdef __DEBUG_DETAIL_INFO
    c_end=clock();
    cout << "[Info]clearImg:"<< difftime(c_end,c_start) <<"ms"<<endl;
#endif
}

void multiFrameBlender::feed(const Mat &img){


#ifdef __DEBUG_DETAIL_INFO
    clock_t c_start,c_end;
    c_start=clock();
#endif

    // Create the source image Laplacian pyramid
    Mat img_with_border;
    borderType &border=borders[feed_count_];

    copyMakeBorder(img, img_with_border, border.top, border.bottom, border.left, border.right,
                   BORDER_REFLECT);
#ifdef __DEBUG_DETAIL_INFO
    c_end=clock();
    cout << "[Info]feed1:"<< difftime(c_end,c_start) <<"ms"<<endl;
    c_start=clock();
#endif

    vector<Mat> src_pyr_laplace;
//    resize(img_with_border,img_with_border,Size(736,480));

#ifndef __NOT_USE_OCL_LIB
    if (useGpu) createLaplacePyrGpu(img_with_border, num_bands_, src_pyr_laplace);
    else createLaplacePyr(img_with_border, num_bands_, src_pyr_laplace);
#else
    createLaplacePyr(img_with_border, num_bands_, src_pyr_laplace);
#endif

#ifdef __DEBUG_DETAIL_INFO
    c_end=clock();
    cout << "[Info]feed2:"<< difftime(c_end,c_start) <<"ms"<<endl;
    c_start=clock();
#endif

    vector<Mat> &weight_pyr_gauss=imgs_mask_pyr_gauss[feed_count_];

#ifdef __DEBUG_DETAIL_INFO
    c_end=clock();
    cout << "[Info]feed3:"<< difftime(c_end,c_start) <<"ms"<<endl;
    c_start=clock();
#endif

    int y_tl = border.tl.y - dst_roi_.y;
    int y_br = border.br.y - dst_roi_.y;
    int x_tl = border.tl.x - dst_roi_.x;
    int x_br = border.br.x - dst_roi_.x;

#ifdef __DEBUG_DETAIL_INFO
    c_end=clock();
    cout << "[Info]feed4:"<< difftime(c_end,c_start) <<"ms"<<endl;
    c_start=clock();
#endif

    // Add weighted layer of the source image to the final Laplacian pyramid layer
    for (int i = 0; i <= num_bands_; ++i)
    {
        for (int y = y_tl; y < y_br; ++y)
        {
            int y_ = y - y_tl;
            const Point3_<short>* src_row = src_pyr_laplace[i].ptr<Point3_<short> >(y_);
            Point3_<short>* dst_row = dst_pyr_laplace_[i].ptr<Point3_<short> >(y);
            const float* weight_row = weight_pyr_gauss[i].ptr<float>(y_);

            for (int x = x_tl; x < x_br; ++x)
            {
                int x_ = x - x_tl;
                dst_row[x].x += static_cast<short>(src_row[x_].x * weight_row[x_]);
                dst_row[x].y += static_cast<short>(src_row[x_].y * weight_row[x_]);
                dst_row[x].z += static_cast<short>(src_row[x_].z * weight_row[x_]);

            }
        }
        x_tl /= 2; y_tl /= 2;
        x_br /= 2; y_br /= 2;
    }

    feed_count_++;

#ifdef __DEBUG_DETAIL_INFO
    c_end=clock();

    cout << "[Info]feed5:"<< difftime(c_end,c_start) <<"ms"<<endl;
#endif

}

void multiFrameBlender::bulitMaskPyr(const Mat &mask, Point tl){

    if (mask.type() != CV_8U) return ;

    // Keep source image in memory with small border
    int gap = 3 * (1 << num_bands_);
    Point tl_new(max(dst_roi_.x, tl.x - gap),
                 max(dst_roi_.y, tl.y - gap));
    Point br_new(min(dst_roi_.br().x, tl.x + mask.cols + gap),
                 min(dst_roi_.br().y, tl.y + mask.rows + gap));

    // Ensure coordinates of top-left, bottom-right corners are divided by (1 << num_bands_).
    // After that scale between layers is exactly 2.
    //
    // We do it to avoid interpolation problems when keeping sub-images only. There is no such problem when
    // image is bordered to have size equal to the final image size, but this is too memory hungry approach.
    tl_new.x = dst_roi_.x + (((tl_new.x - dst_roi_.x) >> num_bands_) << num_bands_);
    tl_new.y = dst_roi_.y + (((tl_new.y - dst_roi_.y) >> num_bands_) << num_bands_);

    int width = br_new.x - tl_new.x;
    int height = br_new.y - tl_new.y;
    width += ((1 << num_bands_) - width % (1 << num_bands_)) % (1 << num_bands_);
    height += ((1 << num_bands_) - height % (1 << num_bands_)) % (1 << num_bands_);
    br_new.x = tl_new.x + width;
    br_new.y = tl_new.y + height;

    int dy = max(br_new.y - dst_roi_.br().y, 0);
    int dx = max(br_new.x - dst_roi_.br().x, 0);
    tl_new.x -= dx; br_new.x -= dx;
    tl_new.y -= dy; br_new.y -= dy;

    borderType border;
    border.tl=tl_new;
    border.br=br_new;

    border.top = tl.y - tl_new.y;
    border.left = tl.x - tl_new.x;
    border.bottom = br_new.y - tl.y - mask.rows;
    border.right = br_new.x - tl.x - mask.cols;
    borders.push_back(border);

    // Create the weight map Gaussian pyramid
    Mat weight_map;
    vector<Mat> weight_pyr_gauss(num_bands_ + 1);

    mask.convertTo(weight_map, CV_32F, 1./255.);

    copyMakeBorder(weight_map, weight_pyr_gauss[0], border.top, border.bottom, border.left, border.right, BORDER_CONSTANT);

    for (int i = 0; i < num_bands_; ++i)
        pyrDown(weight_pyr_gauss[i], weight_pyr_gauss[i + 1]);

    int y_tl = tl_new.y - dst_roi_.y;
    int y_br = br_new.y - dst_roi_.y;
    int x_tl = tl_new.x - dst_roi_.x;
    int x_br = br_new.x - dst_roi_.x;

    for (int i = 0; i <= num_bands_; ++i)
    {
        for (int y = y_tl; y < y_br; ++y)
        {
            float* dst_weight_row = dst_band_weights_[i].ptr<float>(y);
            const float* weight_row = weight_pyr_gauss[i].ptr<float>(y - y_tl);

            for (int x = x_tl; x < x_br; ++x)
            {
                dst_weight_row[x] += weight_row[x - x_tl];
            }
         }
        x_tl /= 2; y_tl /= 2;
        x_br /= 2; y_br /= 2;
     }

    imgs_mask_pyr_gauss.push_back(weight_pyr_gauss);
}

void multiFrameBlender::normalizeWeightPyr(){
     for (unsigned int k=0;k < borders.size();k++){
         borderType &border=borders[k];
         int y_tl = border.tl.y - dst_roi_.y;
         int y_br = border.br.y - dst_roi_.y;
         int x_tl = border.tl.x - dst_roi_.x;
         int x_br = border.br.x - dst_roi_.x;
         vector<Mat> &weight_pyr_gauss_s =imgs_mask_pyr_gauss[k];//第k个MASK

         for (int i = 0; i <= num_bands_; i++){
             Mat &dist_weight=dst_band_weights_[i];  //第i层波段
             Mat &weight_pyr_gauss=weight_pyr_gauss_s[i];  //第k个MASK的第i层波段

             for (int y = y_tl; y < y_br; ++y)
             {
                 float* dst_weight_row = dist_weight.ptr<float>(y);
                 float* weight_row = weight_pyr_gauss.ptr<float>(y - y_tl);

                 for (int x = x_tl; x < x_br; ++x)
                 {
                     weight_row[x - x_tl] = weight_row[x - x_tl]/(dst_weight_row[x] + WEIGHT_EPS); //归一化
                 }
             }
             x_tl /= 2; y_tl /= 2;
             x_br /= 2; y_br /= 2;
         }
     }
     //dst_band_weights_.clear();  //这个空间可以释放了
}

//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

#define __USE_OPENCV_PYR

void normalizeUsingWeightMap(const Mat& weight, Mat& src)
{
    if (src.type() != CV_16SC3) return;

    if(weight.type() == CV_32FC1)
    {
        for (int y = 0; y < src.rows; ++y)
        {
            Point3_<short> *row = src.ptr<Point3_<short> >(y);
            const float *weight_row = weight.ptr<float>(y);

            for (int x = 0; x < src.cols; ++x)
            {
                row[x].x = static_cast<short>(row[x].x / (weight_row[x] + WEIGHT_EPS));
                row[x].y = static_cast<short>(row[x].y / (weight_row[x] + WEIGHT_EPS));
                row[x].z = static_cast<short>(row[x].z / (weight_row[x] + WEIGHT_EPS));
            }
        }
    }
}

void createLaplacePyr(const Mat &img, int num_levels, vector<Mat> &pyr)
{
    pyr.resize(num_levels + 1);
    if(img.depth() == CV_8U)
    {
        if(num_levels == 0)
        {
            img.convertTo(pyr[0], CV_16S);
            return;
        }

        Mat downNext;

#ifdef __USE_OPENCV_PYR
        Mat current = img;
        pyrDown(img, downNext);
#else
        Mat imgTemp;
        img.convertTo(imgTemp, CV_16S);

        Mat current =imgTemp;
        myPyrDown(imgTemp, downNext);
#endif
        for(int i = 1; i < num_levels; ++i)
        {

            Mat lvl_down;
#ifdef __USE_OPENCV_PYR
            Mat lvl_up;
            pyrDown(downNext, lvl_down);
            pyrUp(downNext, lvl_up, current.size());
            subtract(current, lvl_up, pyr[i-1], noArray(), CV_16S);
#else
            myPyrDown(downNext,lvl_down);
            myLaplacePyr(downNext,current, pyr[i-1]);
#endif
            current = downNext;
            downNext = lvl_down;
        }

        {
#ifdef __USE_OPENCV_PYR
            Mat lvl_up;
            pyrUp(downNext, lvl_up, current.size());
            subtract(current, lvl_up, pyr[num_levels-1], noArray(), CV_16S);
            downNext.convertTo(pyr[num_levels], CV_16S);
#else
            myLaplacePyr(downNext,current,pyr[num_levels-1]);
            pyr[num_levels]=downNext;
#endif

        }
    }
}

#ifndef __NOT_USE_OCL_LIB
void createLaplacePyrGpu(const Mat &img, int num_levels, std::vector<Mat> &pyr){
    pyr.resize(num_levels + 1);

    vector<oclMat> oclPyr(num_levels + 1);
    oclPyr[0].upload(img);

    for (int i = 0; i < num_levels; ++i)
        ocl::pyrDown(oclPyr[i], oclPyr[i + 1]);

    oclMat tmp;
    for (int i = 0; i < num_levels; ++i)
    {
        ocl::pyrUp(oclPyr[i + 1], tmp);
        ocl::subtract(oclPyr[i], tmp, oclPyr[i]);
        oclPyr[i].download(pyr[i]);
        pyr[i].convertTo(pyr[i],CV_16S);
    }

    oclPyr[num_levels].download(pyr[num_levels]);
    pyr[num_levels].convertTo(pyr[num_levels],CV_16S);
}
#endif

void restoreImageFromLaplacePyr(vector<Mat> &pyr)
{
    if (pyr.empty())
        return;
    Mat tmp;
    for (size_t i = pyr.size() - 1; i > 0; --i)
    {
#ifdef __USE_OPENCV_PYR
        pyrUp(pyr[i], tmp, pyr[i - 1].size());
#else
        myPyrUp(pyr[i],tmp);
#endif
        add(tmp, pyr[i - 1], pyr[i - 1]);
    }
}

#define MATTYPE short  //16是这个类型的

void myPyrUp(const Mat &input,Mat &output){  //都是处理CV_16S格式的。并且只把图像长宽扩大2倍
    int rows= input.rows;
    int cols= input.cols;
    int cols_=cols*3;

    output.create(rows*2,cols*2,CV_16SC3); //三通道
    int lineStep=sizeof(MATTYPE)*3*cols*2;

    for (int i=0;i<rows;i++){
        const MATTYPE *inputRow=input.ptr<MATTYPE>(i);
        MATTYPE *outputRow1=output.ptr<MATTYPE>(2*i);
        MATTYPE *outputRow2=output.ptr<MATTYPE>(2*i+1);

        for (int j=0,k=0;j<cols_;j++,k++){
            outputRow1[k]=inputRow[j];
            outputRow1[++k]=inputRow[j+1];
            outputRow1[++k]=inputRow[j+2];
            ++k;
            outputRow1[k]=inputRow[j];
            outputRow1[++k]=inputRow[++j];
            outputRow1[++k]=inputRow[++j];
        }

        memcpy(outputRow2,outputRow1,lineStep);

    }

}

void myLaplacePyr(const Mat &downInput,const  Mat &source, Mat &outImg){
    int rows=downInput.rows;
    int cols=downInput.cols;
    int cols_=source.cols;
    int rows_=source.rows;

    if ((rows_!=rows*2)||(cols_!=cols*2)){
        throw sysException("Laplace downImg size should be 1/4 of source size");
    }
    outImg.create(rows_,cols_,CV_16SC3);

    int cols__=cols*3;
    for (int i=0;i<rows;i++){
        const MATTYPE *inputRow=downInput.ptr<MATTYPE>(i);
        MATTYPE *outputRow1=outImg.ptr<MATTYPE>(2*i);
        MATTYPE *outputRow2=outImg.ptr<MATTYPE>(2*i+1);
        const MATTYPE *sourceRow1=source.ptr<MATTYPE>(2*i);
        const MATTYPE *sourceRow2=source.ptr<MATTYPE>(2*i+1);

        for (int j=0,k=0;j<cols__;j+=3,k+=3){
            outputRow1[k]=sourceRow1[k]-inputRow[j];
            outputRow1[k+1]=sourceRow1[k+1]-inputRow[j+1];
            outputRow1[k+2]=sourceRow1[k+2]-inputRow[j+2];

            outputRow2[k]=sourceRow2[k]-inputRow[j];
            outputRow2[k+1]=sourceRow2[k+1]-inputRow[j+1];
            outputRow2[k+2]=sourceRow2[k+2]-inputRow[j+2];

            k+=3;
            outputRow1[k]=sourceRow1[k]-inputRow[j];
            outputRow1[k+1]=sourceRow1[k+1]-inputRow[j+1];
            outputRow1[k+2]=sourceRow1[k+2]-inputRow[j+2];

            outputRow2[k]=sourceRow2[k]-inputRow[j];
            outputRow2[k+1]=sourceRow2[k+1]-inputRow[j+1];
            outputRow2[k+2]=sourceRow2[k+2]-inputRow[j+2];
        }

    }

}

void myPyrDown(const Mat &input, Mat &output){  //都是处理CV_16S格式的。并且只把图像长宽缩小两倍2倍，长宽不是偶数的不处理
    int rows= input.rows;
    int cols= input.cols;

    if ((rows%2!=0)||(cols%2!=0)){   //一定要整除才可以
        throw sysException("Rows or cols of Pyrimid Image is not 2^n!");
    }else{
        output.create(rows/2,cols/2,CV_16SC3);
    }

    int rows_=rows/2;
    int cols_=cols/2 * 3;
    for (int i=0;i<rows_;i++){
        const MATTYPE *inputRow=input.ptr<MATTYPE>(i<<1);
        MATTYPE *outputRow=output.ptr<MATTYPE>(i);

        for (int j=0,k=0;k<cols_;k++){
            outputRow[k]=inputRow[j];
            outputRow[++k]=inputRow[++j];
            outputRow[++k]=inputRow[++j];
            j+=4; // 这里加4不是3哦
        }
    }
}

#undef MATTYPE
