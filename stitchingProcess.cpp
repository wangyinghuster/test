#include "stitchingProcess.h"
#include <sstream>

stitchingProcess::stitchingProcess()
    : videoNum(0),
      videos(NULL),
      outCols(0),
      outRows(0),
      isPrepared(false)
{
    fGnerator=new SIFTfeaturesGenerator;  //特征生成描述子
    fMatch=new SIFTfeaturesMatcher; //特征匹配描述子
    fWarper = new CylindricalKRWarper;//PlaneKRWarper;//parallaxWarper;////parallaxWarper;  //CylindricalKRWarper   //new //warp描述子	sFinder=new DPSeamFinder; //DPSeamFinder;     //拼接缝寻找
    sFinder = new overlapSeamFinder;//overlapSeamFinder;//myCVdpSeamFinder;
	fBlender=new directBlender;//weightFrameBlender;// multiFrameBlender;//directBlender  //融合

	inputPtr=&stitchingProcess::inputVideoBeforePrepare;   //给函数指针附上初值
}

stitchingProcess::~stitchingProcess()
{
    delete fGnerator;
    delete fMatch;
    delete fWarper;
    delete sFinder;
    delete fBlender;

    if (videoNum!=0)       //代表已经初始化过videoInput了。
    {
        for (int i=0;i<videoNum;++i)  
			if (vIdx[i]!=-1)  //某些区域之前释放过
				delete videos[i];
        delete [] videos;
    }
}


void stitchingProcess::generateVideoInput(int num)
{

    if (num<=0) throw sysException("Number of videos should be more than one!");
    
	//idxMap初始化，否则inputVideo不可用
    idxMap.create(1,num,CV_8UC1);
    vIdx=idxMap.ptr<char>(0);

    videos=(videoInput **)(new videoInput *[num]);
    for (int i=0;i<num;i++)
    {
        videoInput * &vptr=videos[i]; 
        //vptr=(videoInput *)(new deviceAVGVideoInput(i));
		vptr=(videoInput *)(new deviceMOGVideoInput(i));
        vIdx[i]=(char)i;
    }
    videoNum=num;

}

//以下是这个函数的两个版本
 void stitchingProcess::inputVideoBeforePrepare(int idx, Mat &img)
{

	__debug(cout << "[Info]Update frame ...[" << videos[vIdx[idx]]->getVideoName()<<"]...";)

    videos[idx]->update(img);  
    videos[idx]->updateBackground();

	__debug(cout << "Done!"<< endl;)

}

void stitchingProcess::inputVideoAfterPrepare(int idx, Mat &img)
{
    if (vIdx[idx]!=-1)
    {
        videos[idx]->update(img);  
    }
}

void stitchingProcess::updateIdx(vector<bool> &flags)
{
	char idxTemp=0;
	for (int i= 0;i<videoNum;++i)
	{
		if (vIdx[i]!=-1)
		{
			if (flags[vIdx[i]]) 
			{
				vIdx[i]=(char)idxTemp;
				++idxTemp;
			}
			else vIdx[i]=-1;

		}
	}
}


void stitchingProcess::prepare(Mat &imgEmptyOut)
{
    imgSet.clear();
    imgFeature.clear();

	__debug(cout<<"[Info]Generate background and find features..."<<endl;)
    for (int i=0;i<videoNum;++i)  //这里由于没有对idxMap进行操作，所以没啥
    {
        Mat img;
        imgFeatures feature;
        videos[i]->getBackground(img);           //用背景进行拼接参数的计算
        fGnerator->detectFeature(img,feature);

        imgSet.push_back(img);
        imgFeature.push_back(feature);
    }

	__debug(cout<<"[Info]Matching imgs..."<<endl;)
    fMatch->buildMatch(imgSet,imgFeature,imgMatchInfo);

	__debug(cout<<"[Info]Find match sets..."<<endl;)
	Mat temp;
    fMatch->findSeperatedMatchSets(temp);

	__debug(cout<<"[Info]Adjusting matching relationship..."<<endl;)
    vector<bool> flags;
    fMatch->findLargestSets(imgSet,imgFeature,imgMatchInfo,flags);  //对img进行裁剪
 
	//这里需要重新修改idxMap;
    updateIdx(flags);

	__debug(cout<<"[Info]Prepare to warp..."<<endl;)

    fWarper->prepare(imgSet,imgFeature,imgMatchInfo,imgMaskWarp,topleft,flags);

    __debug(
    for (unsigned int i=0;i<imgMaskWarp.size();++i)
    {
        stringstream ss;
        ss << "imgMaskWarp" << i <<".jpg";
        string s;
        ss >> s;
        cvSaveImage(s.c_str(),& IplImage(imgMaskWarp[i]));
    })

    //修改flag
    updateIdx(flags);


	//更新videos[]数组指针指向的内存区域，释放拼接中不需要的videoInput的空间
	for (int i=0;i<videoNum;++i)    //这里需要通过idxMap进行寻找
		if (vIdx[i]==-1)  delete videos[i];

	__debug(cout<<"[Info]Warping imgs..."<<endl;)
    fWarper->doWarp(imgSet,imgWarp);

    __debug(
    for (unsigned int i=0;i<imgWarp.size();++i)
    {
        stringstream ss;
        ss << "imgWarp" << i <<".jpg";
        string s;
        ss >> s;
        cvSaveImage(s.c_str(),& IplImage(imgWarp[i]));
    })

	__debug(cout<<"[Info]Find seam for blend..."<<endl;)
    sFinder->findSeam(imgWarp,imgMaskWarp,imgSeamMask,topleft);

    __debug(
    for (unsigned int i=0;i<imgSeamMask.size();++i)
    {
        stringstream ss;
        ss << "imgSeamMask" << i <<".jpg";
        string s;
        ss >> s;
        cvSaveImage(s.c_str(),& IplImage(imgSeamMask[i]));
    })

	__debug(cout<<"[Info]Prepare to blend..."<<endl;)
    fBlender->prepare(imgSeamMask,imgMaskWarp,topleft);

	__debug(cout<<"[Info]Blending imgs..."<<endl;)
    fBlender->doBlend(imgWarp,topleft,imgEmptyOut);

    isPrepared=true;  //更新状态

	inputPtr=&stitchingProcess::inputVideoAfterPrepare;  //更新input的指针

    __debug(stringstream ss;
    ss <<"Prepare_stitching_result_" << imgEmptyOut.rows <<"*"<< imgEmptyOut.cols;
    string s;
    ss >> s;)
    __debug(namedWindow(s.c_str(),CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);)
    __debug(cvShowImage(s.c_str(),& IplImage(imgEmptyOut));)
    __debug(waitKey(0);)
    //imgEmptyOut=Mat::zeros(1080,1920,CV_8UC3);
	__debug(cout<<"[Info]Done!"<<endl;)
}

void stitchingProcess::stitch(Mat &imgOut)
{
    if (!isPrepared) throw sysException("Please prepare for stitching first!");

	__debug(cout<<"[Info]Stitching...";)
    imgSet.clear();
	vector<Mat> fMask;
	vector<Mat> imgForeWarp;
    for (int i=0;i<videoNum;++i)    //这里需要通过idxMap进行寻找
    {
        if (vIdx[i]!=-1)
        {
            Mat img;
            videos[i]->getFrame(img);
            imgSet.push_back(img);
			videos[i]->getForeMask(img);
			fMask.push_back(img);
        }
    }

	__debug(cout<<"Warping...";)
    fWarper->doWarp(imgSet,imgWarp);
	fWarper->doWarp(fMask,imgForeWarp);

	__debug(
    for (unsigned int i=0;i<imgForeWarp.size();++i)
    {
        stringstream ss;
        ss << "imgForeWarp_0" << i <<".jpg";
        string s;
        ss >> s;
        cvSaveImage(s.c_str(),& IplImage(fMask[i]));
    })
	__debug(cout<<"Adjusting...";)
	fBlender->adjustForground(imgWarp,imgForeWarp,topleft);
	__debug(
    for (unsigned int i=0;i<imgForeWarp.size();++i)
    {
        stringstream ss;
        ss << "imgForeWarp" << i <<".jpg";
        string s;
        ss >> s;
        cvSaveImage(s.c_str(),& IplImage(imgForeWarp[i]));
    })

	__debug(cout<<"Blending...";)
    fBlender->doBlend(imgWarp,topleft,imgOut);

    __debug(stringstream ss;
    ss <<"Stitching_result_"<< imgOut.rows <<"*"<< imgOut.cols;
    string s;
    ss >> s;)
    __debug(namedWindow(s.c_str(),CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);)
    __debug(cvShowImage(s.c_str(),& IplImage(imgOut));)
    __debug(waitKey(1);)
    //imgOut=Mat::zeros(1080,1920,CV_8UC3);
	__debug(cout<<"Done!"<<endl;)




}
