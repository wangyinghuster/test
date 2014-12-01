#include "opCVseamFinder.h"

void graphCutSeamFinder::findSeam(vector<Mat> &warpImages,vector<Mat> &warpMask,vector<Mat> &seamMsk,vector<Point> &topleft){

    Ptr<SeamFinder> seamFinder;
    if (useGrad)
        seamFinder = new GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR_GRAD);  //这个速度最慢，慢的都死了（没有resize，实测，5幅图23min），resize之后比较快（实测，5幅图3s）
    else
        seamFinder = new GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);


    int xmax=0,ymax=0;
    for (unsigned int i=0;i<warpMask.size();i++)
    {
        Point &p=topleft[i];
        int xT=p.x+warpMask[i].cols;
        int yT=p.y+warpMask[i].rows;
        xmax=xmax>xT?xmax:xT;
        ymax=ymax>yT?ymax:yT;
    }


    //计算resize参数的大小。这里需要计算一下resize，不然的话实在是太慢了。
    double seamMegapix=0.1; //这个参数，自己看看能不能改
    double seamScale;
    seamScale=sqrt(seamMegapix * 1e6 / (ymax*xmax));
    seamScale=1.0<seamScale?1.0:seamScale;  //计算resize参数，不让放大，只让缩小

    vector<Mat> imgF(warpMask.size());
    for (unsigned int i=0;i<warpMask.size();i++){
        warpImages[i].convertTo(imgF[i],CV_32F); //这里需要转换图像深度
        seamMsk.push_back(warpMask[i].clone());   //这里需要这么复制，否则seamMask就和imgMask共用一块内存区域。

        resize(imgF[i],imgF[i],Size(),seamScale,seamScale); //变换大小
        resize(seamMsk[i],seamMsk[i],Size(),seamScale,seamScale); //变换大小

    }

    seamFinder->find(imgF,topleft,seamMsk);   //这个会超出内存大小，所以还是得来看看

    for (unsigned int i=0;i<warpMask.size();i++){
        resize(seamMsk[i],seamMsk[i],warpMask[i].size()); //大小换回来	
    }

}

void DPSeamFinder::findSeam(vector<Mat> &warpImages,vector<Mat> &warpMask,vector<Mat> &seamMsk,vector<Point> &topleft){

    Ptr<SeamFinder> seamFinder;
    if (useGrad)
        seamFinder = new DpSeamFinder(DpSeamFinder::COLOR_GRAD);  //Dp的速度还是蛮快的
    else
        seamFinder = new DpSeamFinder(DpSeamFinder::COLOR);

    vector<Mat> imgF;
    imgF.resize(warpImages.size());
    for (unsigned int i=0;i<warpMask.size();i++)
	{
        warpImages[i].convertTo(imgF[i],CV_32F); //这里需要转换图像深度
        seamMsk.push_back(warpMask[i].clone());   //这里需要这么复制，否则seamMask就和imgMask共用一块内存区域。
    }

    seamFinder->find(imgF,topleft,seamMsk);

}
