#include "opCVseamFinder.h"

void graphCutSeamFinder::findSeam(vector<Mat> &warpImages,vector<Mat> &warpMask,vector<Mat> &seamMsk,vector<Point> &topleft){

    Ptr<SeamFinder> seamFinder;
    if (useGrad)
        seamFinder = new GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR_GRAD);  //����ٶ����������Ķ����ˣ�û��resize��ʵ�⣬5��ͼ23min����resize֮��ȽϿ죨ʵ�⣬5��ͼ3s��
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


    //����resize�����Ĵ�С��������Ҫ����һ��resize����Ȼ�Ļ�ʵ����̫���ˡ�
    double seamMegapix=0.1; //����������Լ������ܲ��ܸ�
    double seamScale;
    seamScale=sqrt(seamMegapix * 1e6 / (ymax*xmax));
    seamScale=1.0<seamScale?1.0:seamScale;  //����resize���������÷Ŵ�ֻ����С

    vector<Mat> imgF(warpMask.size());
    for (unsigned int i=0;i<warpMask.size();i++){
        warpImages[i].convertTo(imgF[i],CV_32F); //������Ҫת��ͼ�����
        seamMsk.push_back(warpMask[i].clone());   //������Ҫ��ô���ƣ�����seamMask�ͺ�imgMask����һ���ڴ�����

        resize(imgF[i],imgF[i],Size(),seamScale,seamScale); //�任��С
        resize(seamMsk[i],seamMsk[i],Size(),seamScale,seamScale); //�任��С

    }

    seamFinder->find(imgF,topleft,seamMsk);   //����ᳬ���ڴ��С�����Ի��ǵ�������

    for (unsigned int i=0;i<warpMask.size();i++){
        resize(seamMsk[i],seamMsk[i],warpMask[i].size()); //��С������	
    }

}

void DPSeamFinder::findSeam(vector<Mat> &warpImages,vector<Mat> &warpMask,vector<Mat> &seamMsk,vector<Point> &topleft){

    Ptr<SeamFinder> seamFinder;
    if (useGrad)
        seamFinder = new DpSeamFinder(DpSeamFinder::COLOR_GRAD);  //Dp���ٶȻ��������
    else
        seamFinder = new DpSeamFinder(DpSeamFinder::COLOR);

    vector<Mat> imgF;
    imgF.resize(warpImages.size());
    for (unsigned int i=0;i<warpMask.size();i++)
	{
        warpImages[i].convertTo(imgF[i],CV_32F); //������Ҫת��ͼ�����
        seamMsk.push_back(warpMask[i].clone());   //������Ҫ��ô���ƣ�����seamMask�ͺ�imgMask����һ���ڴ�����
    }

    seamFinder->find(imgF,topleft,seamMsk);

}
