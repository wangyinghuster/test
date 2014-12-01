#include "featuresGenerator.h"

int featuresGenerator::colsMax=512;  //�ֿ���������Ŀ��С
int featuresGenerator::rowsMax=512;  //�ֿ���������Ŀ��С

featuresGenerator::featuresGenerator()
    :method("NULL")
{
}

featuresGenerator::~featuresGenerator(){}  //�����������������ṩһ������

/*
void featuresGenerator::detectFeature(Mat &image, imgFeatures &feature)  //����汾��������ȡ�����ģ��޷�Ӧ���ڴ治������
{

    feature.method=method;
    Ptr<FeatureDetector> detector= FeatureDetector::create(method);

    detector->detect(image,feature.backGroundPoint);

    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(method);
    extractor->compute(image,feature.backGroundPoint,feature.backGroundFeature);

}
*/

void featuresGenerator::detectFeature(Mat &image,imgFeatures &feature)  //����汾��ͼƬ���зָ�ֿ���ȡ������Ӧ���ڴ治��
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
		if (height>rowsMax) height=rowsMax;  //��ֹԽ��

		int left=0;
		while(left<image.cols)
		{  
			 int width=image.cols-left;
			 if (width>colsMax) width=colsMax;   //��ֹԽ��

             Mat imgTemp=image(Rect(left,top,width,height)).clone();

			 //���������
			 vector<KeyPoint> bgPoint;
			 Mat bgFeature;
			 detector->detect(imgTemp,bgPoint);
	         extractor->compute(imgTemp,bgPoint,bgFeature);

			 //����ת��������������KeyPoint����֪���ǲ���ֻҪת������Ϳ����ˣ�
             for (unsigned int i=0;i<bgPoint.size();++i)
			 {  
                  bgPoint[i].pt.x+=left;
                  bgPoint[i].pt.y+=top;
			 }
             fgbPoint.insert(fgbPoint.end(),bgPoint.begin(),bgPoint.end());  //����ϲ���������������

			 //�ȱ�������������������ͳһ�ϲ�
			 bgFeatureSets.push_back(bgFeature);

		     left+=colsMax;
		}

		top+=rowsMax;
	}
	
	//��ʼ�ϲ�������Mat;
	int allRows=0;
	
	int u=0;
	try{
		while(bgFeatureSets[u].cols==0) u++;
	}catch(...){
		throw sysException("[Failed]Image has no features!!!");  //������һ��������û���ҵ������ߵ�����ᷢ��vectorԽ��,invalidʲô��
	}
	
	int allCols=bgFeatureSets[u].cols;

	for (unsigned int i=0;i<bgFeatureSets.size();++i)  //ͳ���ܹ�������Ҳ�������������Ŀ������Ϊ������������
	{
        allRows+=bgFeatureSets[i].rows;
	}

	fgbFeat=Mat::zeros(allRows,allCols,bgFeatureSets[u].type());//��������
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
    //Flann ������ҪCV_32F���ͣ���ORB���������ò��ǣ�������Ҫת��������
    feature.backGroundFeature.convertTo(feature.backGroundFeature, CV_32F);
}
