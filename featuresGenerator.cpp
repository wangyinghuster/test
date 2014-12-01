#include "featuresGenerator.h"

int featuresGenerator::colsMax=512;  //分块检测特征点的块大小
int featuresGenerator::rowsMax=512;  //分块检测特征点的块大小

featuresGenerator::featuresGenerator()
    :method("NULL")
{
}

featuresGenerator::~featuresGenerator(){}  //纯虚析构函数必须提供一个定义

/*
void featuresGenerator::detectFeature(Mat &image, imgFeatures &feature)  //这个版本是整个求取特征的，无法应对内存不足问题
{

    feature.method=method;
    Ptr<FeatureDetector> detector= FeatureDetector::create(method);

    detector->detect(image,feature.backGroundPoint);

    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(method);
    extractor->compute(image,feature.backGroundPoint,feature.backGroundFeature);

}
*/

void featuresGenerator::detectFeature(Mat &image,imgFeatures &feature)  //这个版本将图片进行分割，分块求取，用来应对内存不足
{
	feature.method=method;

	vector<KeyPoint> &fgbPoint=feature.backGroundPoint;
    Mat &fgbFeat=feature.backGroundFeature;

	Ptr<FeatureDetector> detector= FeatureDetector::create(method);
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(method);

	vector<Mat> bgFeatureSets;
	int top=0;
    while(top<image.rows)
	{
		int height=image.rows-top;
		if (height>rowsMax) height=rowsMax;  //防止越界

		int left=0;
		while(left<image.cols)
		{  
			 int width=image.cols-left;
			 if (width>colsMax) width=colsMax;   //防止越界

             Mat imgTemp=image(Rect(left,top,width,height)).clone();

			 //检测特征点
			 vector<KeyPoint> bgPoint;
			 Mat bgFeature;
			 detector->detect(imgTemp,bgPoint);
	         extractor->compute(imgTemp,bgPoint,bgFeature);

			 //坐标转换，由于类型是KeyPoint，不知道是不是只要转换坐标就可以了？
             for (unsigned int i=0;i<bgPoint.size();++i)
			 {  
                  bgPoint[i].pt.x+=left;
                  bgPoint[i].pt.y+=top;
			 }
             fgbPoint.insert(fgbPoint.end(),bgPoint.begin(),bgPoint.end());  //这里合并这两个特征向量

			 //先保存这个特征，到最后在统一合并
			 bgFeatureSets.push_back(bgFeature);

		     left+=colsMax;
		}

		top+=rowsMax;
	}
	
	//开始合并特征的Mat;
	int allRows=0;
	
	int u=0;
	try{
		while(bgFeatureSets[u].cols==0) u++;
	}catch(...){
		throw sysException("[Failed]Image has no features!!!");  //这里是一个特征都没有找到，会走到这里。会发生vector越界,invalid什么的
	}
	
	int allCols=bgFeatureSets[u].cols;

	for (unsigned int i=0;i<bgFeatureSets.size();++i)  //统计总共行数，也就是特征点的数目，列数为特征向量长度
	{
        allRows+=bgFeatureSets[i].rows;
	}

	fgbFeat=Mat::zeros(allRows,allCols,bgFeatureSets[u].type());//创建矩阵
	int rowPtr=0;
	for (unsigned int i=0;i<bgFeatureSets.size();++i)
	{
		if (bgFeatureSets[i].rows==0) continue;
		Mat fgbTmp=fgbFeat(Rect(0,rowPtr,allCols,bgFeatureSets[i].rows));
        bgFeatureSets[i].copyTo(fgbTmp); //Here;
		rowPtr+=bgFeatureSets[i].rows;
	}
	//__debug(cout <<fgbFeat <<endl;);
}


void ORBfeaturesGenerator::detectFeature(Mat &image, imgFeatures &feature)
{
    featuresGenerator::detectFeature(image,feature);
    //Flann 计算需要CV_32F类型，而ORB的描述正好不是，所以需要转换。。。
    feature.backGroundFeature.convertTo(feature.backGroundFeature, CV_32F);
}
