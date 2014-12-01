#include "parallaxWarper.h"
//#include "opencv2/core/core.hpp"
#include "kdtree.h"
#include <strstream>
#include <queue>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <string>
#include <limits>
#include <fstream>

#define M_PI    3.14159265358979323846

using namespace cv;
using namespace std;


Scalar colorScale(int idx,int allNum)  //����һ���㷨��ɫ�ʵ����ֶ����
{
	int red;
	int green;
	int blue;
	allNum--; //idx��0��ʼ�ġ�

	red=(idx%2)?(allNum-(idx-1)/2):(idx/2);
	red=red*255/allNum;

	green=(idx/2)%2?(allNum-(idx)/2):(idx/2);
	green=green*255/allNum;

	blue=(idx/4)%2?(allNum-(idx)/2):(idx/2);
	blue=blue*255/allNum;

	Scalar color(red,green,blue);
	return color;
}


parallaxWarper::parallaxWarper()
{

}

/************************************************************************/
/* ͶӰ׼���׶���������������ͶӰ����H������ƥ������wrapԤ����  */
/************************************************************************/

void parallaxWarper::prepare(vector<Mat> &imgSet, vector<imgFeatures> &imgF,
							 vector<mLog> &matchInfoIn, vector<Mat> &imgMaskWarpOut,
							 vector<Point> &topleftOut, vector<bool> &videoFlag)
{
	 mtNode* head = buildMatchTree(imgSet, imgF ,matchInfoIn, videoFlag);
	 vector<hMap> hmapOut;
	 calculateLocalH(imgSet, imgF, matchInfoIn, head,hmapOut);	 
	 releaseMatchTree(head);
	 cout << "here";
	 preWarp(imgSet, hmapOut, imgMaskWarpOut, topleftOut);
}


void parallaxWarper::findHSet(imgFeatures &imgFsrc,imgFeatures &imgFdst,
                              vector<DMatch> &matchIn,  //����Ҫ��ƥ�䣬DMatch��trainIndxӦ����pair��first��idx��
                              vector<vector<int> > &keyPointIdxOut,
                              vector<pair<Mat, Mat> > &HSetsOut)
{
    vector<KeyPoint> &dstPoint=imgFdst.backGroundPoint;   //first����Ϊdst�õ�
    vector<KeyPoint> &srcPoint=imgFsrc.backGroundPoint;

    //ѭ����ʼ��
    //�����vec<KeyP>ת����Mat,�������ƥ��ĵ�
    vector<Point2f> src;
    vector<Point2f> dst;
    vector<int> idxTemp;
	src.resize(matchIn.size());
	dst.resize(matchIn.size());
	idxTemp.resize(matchIn.size());
    for (unsigned int i=0;i<matchIn.size();++i)
    {
        DMatch &dm=matchIn[i];                             //����Ҫ��ƥ�䣬DMatch��trainIndxӦ����pair��first��idx��
        src[i]=srcPoint[dm.queryIdx].pt;
        dst[i]=dstPoint[dm.trainIdx].pt;
        idxTemp[i]=i;
    }
    keyPointIdxOut.clear();
    HSetsOut.clear();
    //ѭ����ʼ
    vector<uchar> srcMask;
    int inliners=0;
    while(idxTemp.size()>(unsigned int)10){   //ʣ��ĵ�����10�����over

        inliners=0;
        Mat homo=findHomography(src,dst,srcMask,CV_RANSAC,1.0);  //�����ǲ���Ҫ����ֵ���õ�Сһ�㣿

        vector<Point2f> srcNext;
        vector<Point2f> dstNext;
        vector<int> idxTempNext;
        vector<int> numIdx;

        for (unsigned int i=0;i<srcMask.size();++i)
        {
            if (srcMask[i])  //������ڵ�
            {
                numIdx.push_back(idxTemp[i]);
                inliners++;
            }
            else   //��������ڵ�
            {
                cout <<srcNext.size()<<"<"<< srcNext.capacity()<<"," ;
                srcNext.push_back(src[i]);
                dstNext.push_back(dst[i]);
                idxTempNext.push_back(idxTemp[i]);
            }
        }

        if (inliners>10){            //�ڵ�������10���ű���
            keyPointIdxOut.push_back(numIdx);
            Mat homoRev=homo.inv();
            HSetsOut.push_back(make_pair(homo,homoRev)); //������һ��ľ���
        }
        else break;   //����10��Ҳover

        srcMask.clear();  //���
        src=srcNext;
        dst=dstNext;
        idxTemp=idxTempNext;
    }
    cout << "here1";
    //�������뻭һ��ͼ�������������ʲô���
    //�ƺ�û��ԭͼ�����˰�������
}

bool parallaxWarper::isCross(Point2f s1,Point2f dirct1,Point2f d1,Point2f d2){//�������һЩ��ʱ��arcos�����ˣ�רע��ȷ��
    dirct1=dirct1-s1; //��֪��������û������
    d1=d1-s1;
    d2=d2-s1;

    double d=acos(dirct1.x/sqrt(dirct1.x*dirct1.x+dirct1.y*dirct1.y));
    if (dirct1.y<0) d=2*M_PI-d;

    double a1=acos(d1.x/sqrt(d1.x*d1.x+d1.y*d1.y));
    if (d1.y<0) a1=2*M_PI-a1;

    double a2=acos(d2.x/sqrt(d2.x*d2.x+d2.y*d2.y));
    if (d2.y<0) a2=2*M_PI-a2;

    if (abs(a1-a2)<M_PI)
    {
        if (a1<a2)
            return ((a1<d)&&(d<a2));
        else if (a1>a2)
            return ((a2<d)&&(d<a1));
        else return false;

    }else if (abs(a1-a2)>M_PI)
    {

        if (a1<a2)
            return ((d<a1)||(d>a2));
        else if (a1>a2)
            return ((d<a2)||(d>a1));
        else return false;

    }else return false;

}

void parallaxWarper::calculateLocalH(vector<Mat> &imgSet, vector<imgFeatures> &imgF,
                                     vector<mLog> &matchInfoIn, mtNode *head,
                                     vector<hMap> &hmap_list_out)
{
    queue<mtNode *> nodeQueue;
    vector<hMap> &hListOut =hmap_list_out;
    hListOut.clear();
    hListOut.resize(imgSet.size(),hMap()); //��ʼ��

    //������Ҫ��ʼ��ѭ����ʼ����;    
	//��Ӧ��pushͷ�ڵ㣬����pushͷ���ĺ���,��Ϊͷ�ڵ��hMapֱ�ӳ�ʼ����;
    mtNode *sp=head->son;
	while(sp){  //������ȶ���
		nodeQueue.push(sp);
		sp=sp->brother;
	}
	//��ʼ��ͷ�ڵ��hMap
    int rootId=head->nowImgIdx; //ͷ�������������ָ���Լ�������ֵ��
	__debug(cout<<"[Info]Buliding Hmap of root img "<<rootId<<"..."<<endl;)
    hMap &rootH=hListOut[rootId];
	Mat iden=Mat::zeros(3,3,CV_32FC1);
	setIdentity(iden); //���óɵ�λ��
	int rCol=imgSet[rootId].cols/GridSize+1;
	int rRow=imgSet[rootId].rows/GridSize+1;
	rootH.creat(rRow,rCol);
	for (int i=0;i<rRow;++i)
		for (int j=0;j<rCol;++j)
			rootH.modify(j,i,iden);  //�õ�λ���ʼ��hMap;  

    //�ö��з�ʽ������ȷ�����
    while(!nodeQueue.empty()){
         mtNode * nowNode=nodeQueue.front();nodeQueue.pop();
         mtNode * pt = nowNode->son;
         while(pt){  //������ȶ���
             nodeQueue.push(pt);
             pt=pt->brother;
         }

         mLog &mii=matchInfoIn[nowNode->mLogIdx];
         int nowIdx=mii.queryInx;
         int fatIdx=mii.trainInx;

         cout <<nowNode->fatherImgIdx<<fatIdx<<nowNode->nowImgIdx<<nowIdx<<endl;

         //��ʼ����
         vector<vector<int>> kpSet;
         vector<pair<Mat,Mat>> hSet; //���hSet�������û����
		 __debug(cout<<"[Info]Clusting Feature Point of img "<<nowIdx<<" ..."<<endl;)
         findHSet(imgF[nowIdx],imgF[fatIdx],mii.matchPointIndex,kpSet,hSet);
		 cout << "here";
         hSet.clear(); //����������������

         //�����������pointֵ���ݣ���������Ӧ��KDtress;
		 __debug(cout<<"[Info]Buliding KDtree of each class in img "<<nowIdx<<" ..."<<endl;)
         vector<KdTree> pointKDTree; //���Ӧ��KDTree
         vector<Mat> homoClass;  //���Ӧ��homography����

         vector<DMatch> &dmah= mii.matchPointIndex;
         vector<KeyPoint> &dstPt=imgF[fatIdx].backGroundPoint;   //first����Ϊdst�õ�
         vector<KeyPoint> &srcPt=imgF[nowIdx].backGroundPoint;

         for (unsigned int i=0;i<kpSet.size();++i)
         {
             Mat pointSon;
             Mat pointFat;
             vector<int> &kpList=kpSet[i];
             pointSon.create(kpList.size(),2,CV_32FC1);
             pointFat.create(kpList.size(),2,CV_32FC1);
             for (unsigned int j=0;j<kpList.size();++j)
             {
                 Point s1=dstPt[dmah[kpList[j]].trainIdx].pt; //��ͼ��ĵ�
                 Point s2=srcPt[dmah[kpList[j]].queryIdx].pt; //��ͼ��ĵ�
                 float *daSon=pointSon.ptr<float>(j);
                 float *daFat=pointFat.ptr<float>(j);
                 daSon[0]=static_cast<float>(s2.x);
                 daSon[1]=static_cast<float>(s2.y);
                 daFat[0]=static_cast<float>(s1.x);
                 daFat[1]=static_cast<float>(s1.y);
             }
             vector<uchar> mask;
             Mat homo=findHomography(pointSon,pointFat,mask,CV_RANSAC,3.0);  //�������֮�м���HomographyMatrix
             //������Ҫת��homo�����ͣ�
			 Mat ho;
			 homo.convertTo(ho,CV_32FC1);
             homoClass.push_back(ho);  //����homo����
             KdTree classTree(pointSon,true);  //��ǰ�����KDTree
             pointKDTree.push_back(classTree);//����KDTree;
         }

         //����HMap��ֵ....
         //����Ҫ���HMap�Ĵ�С��
         int cols=imgSet[nowIdx].cols;
         int rows=imgSet[nowIdx].rows;
         int Hcols=(cols/GridSize)+1;
         int Hrows=(rows/GridSize)+1;
		 hMap &hStoF=hListOut[nowIdx];  //����hStoF������Ϊ�˽�ʡ�ڴ棬����ֱ����hList�м�ȡ���������д���
         hStoF.creat(Hrows,Hcols);  //����hMap;

         //Ȼ��ͨ�������HMap
         //���ȼ���H(s->f)�ټ���H(s->ref);
         //����H(s->f)

		 __debug(cout<<"[Info]Building Hmap of img "<<nowIdx<<" to father img "<<fatIdx<<" ..."<<endl;)
         for (int i=0;i<Hrows;++i)
		 {
             for (int j=0;j<Hcols;++j)
			 {
                 //�������ĵ��λ��
                 float c_x=j*GridSize+(GridSize/2.0f);
                 float c_y=i*GridSize+(GridSize/2.0f);
                 float dSum=0.0f;
				 Mat ans=Mat::zeros(3,3,CV_32FC1); //�����ʼ��
                 for (unsigned int k=0;k<pointKDTree.size();++k)
                 {
                     Mat matIn;
                     matIn.create(1,2,CV_32FC1);
                     matIn.at<float>(0,0)=c_x;
                     matIn.at<float>(0,1)=c_y;
                     vector<int> neighborsIdx;
                     pointKDTree[k].findNearest(matIn,1,pointKDTree[k].maxDepth,neighborsIdx);  //��õ���idx
                     const float *dataP=pointKDTree[k].getPoint(neighborsIdx[0]);//��idx���ҵ㣬��õ���һ��float������
                     float d=1.0f/((c_x-dataP[0])*(c_x-dataP[0])+(c_y-dataP[1])*(c_y-dataP[1])+1e-10);  //����ʵ�����õĲ��Ǹ�˹���룬���ǵ������������ڵ����������ģ����ԣ���Ҫ��һ��С��ƫ����
                     dSum+=d;
                     Mat &h=homoClass[k];
                     ans=ans+(h*(double)d);
                 }
                 ans=ans/dSum;
				 hStoF.modify(j,i,ans);
             }
         }

         //�ټ���H(s->ref)
         //����nowͼ���4��������Լ������H(s->f)ӳ��֮�������
		 __debug(cout<<"[Info]Building Hmap of img "<<nowIdx<<" to referance plane..."<<endl;)
         vector<Point2f> vertex;
         vertex.push_back(Point2f(0.0,0.0));    //�����İڷ�˳����Ϊ�����ڵĵ�������һ���߶�
         vertex.push_back(Point2f((float)cols,0.0));
         vertex.push_back(Point2f((float)cols,(float)rows));
         vertex.push_back(Point2f(0.0,(float)rows));

         int pN=hMap::paraNum;
		 float *p=hStoF.data;  //����ָ��data;
         for (vector<Point2f>::iterator it=vertex.begin();it!=vertex.end();++it)
         {
             int xh=(int)it->x/GridSize;
             int yh=(int)it->y/GridSize;
             float * np= p+(yh*Hcols+xh)*pN;
             float x_=it->x;
             float y_=it->y;
             float z_=np[6]*x_+np[7]*y_+1;

             it->x=(np[0]*x_+np[1]*y_+np[2])/z_;
             it->y=(np[3]*x_+np[4]*y_+np[5])/z_;
         } //�������������ͼ������

         vector<Point> bund;  //�߽��ļ���
         int fcols=imgSet[fatIdx].cols/GridSize+1;  //��ͼ��Ĵ�С
         int frows=imgSet[fatIdx].rows/GridSize+1;
         for (int i=1;i<frows-1;++i) {  //������ֹ4�������ص�
             bund.push_back(Point(0,i));
             bund.push_back(Point(fcols-1,i));
         }
         for (int i=0;i<fcols;++i) {
             bund.push_back(Point(i,0));
             bund.push_back(Point(i,frows-1));
         }//���������������������

         //����ʵ���Ͼ��Ǽ���߽����ǲ����ڶ�����ڲ���
         //������򵥵����߷�
         //���ҳ�������С��x
         float xmax=-numeric_limits<float>::max();
         float xmin=numeric_limits<float>::max();
         float ymax=-numeric_limits<float>::max();
         float ymin=numeric_limits<float>::max();
         for (vector<Point2f>::iterator it=vertex.begin();it!=vertex.end();++it)
         {
             float x_=it->x;
             xmax=xmax>x_?xmax:x_;
             xmin=xmin<x_?xmin:x_;
             float y_=it->y;
             ymax=ymax>y_?ymax:y_;
             ymin=ymin<y_?ymin:y_;
         }
         //���ߵ�Ŀ���
         Point2f dst((xmax+xmin)/2.0f,(ymax+ymin)/2.0f);
         vector<Point>::iterator it=bund.begin();
         while(it!=bund.end())
         {
             int count=0;
             Point2f start(it->x*GridSize+GridSize/2.0f,it->y*GridSize+GridSize/2.0f);
             //��һ������
             vector<Point2f>::iterator ip=vertex.begin();
             vector<Point2f>::iterator end=vertex.end()-1;
             while(ip!=end)
             {
                 Point2f p1=(*ip);
                 ip++;
                 Point2f p2=(*ip);
                 if (isCross(start,dst,p1,p2))
                     count++;
             }
			 
             Point2f p1=(*ip);
             Point2f p2=(*vertex.begin());
             if (isCross(start,dst,p1,p2))
                 count++;

			 if (count%2==0) it=bund.erase(it);  //��߽���ż�������㣬�������ı��������ȥ�����
			 else it ++;   //һ�߱���һ��ɾ������Ҫ��ôд
			
         }

         //��ʼ����H(s->ref)
         hMap &fH=hListOut[fatIdx];  //������ӳ�䵽�ο�ƽ��ľ���
         hMap &hStoR=hListOut[nowIdx];  //��ȡhMap������ע����,hStoR��hStoFʵ������ͬһ�������Ĳ�ͬд������
		                                //����ͬһ���ڴ�������������hMap�����и��µ����Բ��������ͻ�Ŀ��ܡ�
										//������������д��ǿ�Ҳ��Ƽ���������������
         Rect img1(0,0,imgSet[fatIdx].cols,imgSet[fatIdx].rows); //img1�ķ�Χ����

         //�޸�hMap��ֵ��
         for (int i=0;i<Hrows;++i){
             for (int j=0;j<Hcols;++j)
             {
                 Mat h2to1=hStoF.at(j,i); //��hStoF�ж�ȡ
                 Mat v=Mat::zeros(3,1,CV_32FC1);  //��������
                 v.at<float>(0,0)=j*GridSize+GridSize/2.0f; //�ŵ�ͼ������x��ֵ
                 v.at<float>(1,0)=i*GridSize+GridSize/2.0f; //�ŵ�ͼ������y��ֵ
                 v.at<float>(2,0)=1.0f;
                 Mat ans=h2to1*v;
                 Point p=Point((int)(ans.at<float>(0,0)/ans.at<float>(2,0)),
                               (int)(ans.at<float>(1,0)/ans.at<float>(2,0)));

                 //cout << p.x <<" "<<p.y<<endl;
                 if (img1.contains(p)) //p���ص�������
                 {
                     Mat h1tof=fH.at(p.x/GridSize,p.y/GridSize);
                     Mat h2tof=h1tof*h2to1;
                     hStoR.modify(j,i,h2tof); //�޸�hStoR;
                 }
                 else
                 {
                     Mat h2tof=Mat::zeros(3,3,CV_32FC1);
                     float sum=0.0;
                     for (vector<Point>::iterator it=bund.begin();it!=bund.end();++it)
                     {
                         float xp=it->x*GridSize+GridSize/2.0f;float yp=it->y*GridSize+GridSize/2.0f;  //��������ת����ȫ������
                         float s=(xp-p.x)*(xp-p.x)+(yp-p.y)*(yp-p.y);
						 s=1.0f/(s+1e-10);  //����Ϊ�˷�ֹ�����
                         sum+=s;
                         Mat h1tof=fH.at(it->x,it->y);
                         h2tof+=s*h1tof*h2to1;
                     }
                     h2tof/=sum;
                     hStoR.modify(j,i,h2tof); //�޸�hStoR;
                 }
             }
         }
    }

    fstream  fileOut("hmap.m",ios::out);   //��һ����Ϊ�����һ��matlab�Ĵ���
    string plotCmd="plot(";
    char clr[]={'b','g','r','c','m','y','k'};
    for (unsigned int w=0;w<hListOut.size();++w)
    {
        hMap &htp=hListOut[w];
        int Hrows=htp.height;
        int Hcols=htp.width;
        fileOut<<"z"<<w<<"=[";
        for (int i=0;i<Hrows;++i){
            for (int j=0;j<Hcols;++j)
            {
                Mat h2to1=htp.at(j,i); //��hStoF�ж�ȡ
                Mat v=Mat::zeros(3,1,CV_32FC1);  //��������
                v.at<float>(0,0)=j*GridSize+GridSize/2.0f; //�ŵ�ͼ������x��ֵ
                v.at<float>(1,0)=i*GridSize+GridSize/2.0f; //�ŵ�ͼ������y��ֵ
                v.at<float>(2,0)=1.0f;
                Mat ans=h2to1*v;
                Point p=Point((int)(ans.at<float>(0,0)/ans.at<float>(2,0)),
                               (int)(ans.at<float>(1,0)/ans.at<float>(2,0)));
                fileOut << p.x <<" "<<p.y<<endl;
            }
        }
        fileOut<<"];"<<endl;
        stringstream ss;
        ss<<plotCmd<<"z"<<w<<"(:,1),"<<"z"<<w<<"(:,2),"<<"'."<<clr[w%7]<<"',";
        ss >> plotCmd;
    }
    plotCmd.erase(plotCmd.end()-1);
    fileOut<<plotCmd<<");"<<endl;
}



/*
void parallaxWarper::prepare(vector<Mat> &imgSet, vector<imgFeatures> &imgF, vector<mLog> &matchInfoIn,
							 vector<Mat> &imgMaskWarpOut, vector<Point> &topleftOut, vector<bool> &videoFlag)
{
	Isodata isodataProcess;
	vector<vector<Cell>> clusters;
	isodataProcess.clustering(imgF,matchInfoIn,clusters);
	isodataProcess.showResult(imgSet,imgF,matchInfoIn,clusters);
}
*/

/************************************************************************/
/* warpԤ����׶Σ�����imgMaskWarp, ӳ���xmap_list��ymap_list       */
/************************************************************************/
void parallaxWarper::preWarp(vector<Mat>&imgSet, vector<hMap> &hmap_list,
							 vector<Mat>&imgMaskWarpOut, vector<Point> &topleftOut)
{
	int videoNum = (int) imgSet.size();

	imgMaskWarpOut.clear();
	xmap_list.clear();
	ymap_list.clear();
	topleftOut.clear();

	imgMaskWarpOut.resize(videoNum);
	xmap_list.resize(videoNum);
	ymap_list.resize(videoNum);
	topleftOut.resize(videoNum);	

	for(int i=0;i<videoNum;i++)
	{
		Rect dst_roi = buildMaps(imgSet[i], hmap_list[i], xmap_list[i], ymap_list[i], imgMaskWarpOut[i]);
		topleftOut[i] = dst_roi.tl();
		/*
		//��warp֮���ͼ��������mask
		imgMaskWarpOut[i].create(dst_roi.height + 1, dst_roi.width + 1, CV_8UC1);
		imgMaskWarpOut[i].setTo(Scalar::all(255));
      
        for(int x=0;x<xmap_list[i].rows; x++)
		{
            for(int y =0; y<xmap_list[i].cols; y++)
			{
				if ((xmap_list[i].at<float>(x,y)==0.f)&&(ymap_list[i].at<float>(x,y)==0.f))
				{
					imgMaskWarpOut[i].at<UCHAR>(x,y)=0;
				}
			}
			cout<<x<<",";
		}*/

		char strname[256];
		sprintf(strname, "warpMask%i.jpg", i);
		cvSaveImage(strname, &IplImage(imgMaskWarpOut[i]));
	}

	for (int k=0;k<imgSet.size();k++)
	{
		cout<<"("<<topleftOut[k].x<<","<<topleftOut[k].y<<")";
	}
}

/************************************************************************/
/* Warp�׶Σ�����imgMask                                               */
/************************************************************************/

void parallaxWarper::doWarp(vector<Mat> &imgSet,vector<Mat>&imgWarpOut)
{
	imgWarpOut.clear();
	imgWarpOut.resize(imgSet.size());
	for( int i=0;i<imgSet.size();i++)
	{
		remap(imgSet[i], imgWarpOut[i], xmap_list[i], ymap_list[i], INTER_NEAREST, BORDER_REFLECT);
	}

}

/************************************************************************/
/* ƥ���ϵ�����������ܺ���                                    */
/************************************************************************/
mtNode* parallaxWarper::buildMatchTree(vector<Mat> &imgSet, vector<imgFeatures> &imgF,
                                       vector<mLog> &matchInfoIn, vector<bool> &videosFlag)
{
	int video_num = (int)imgSet.size();
	int *imgInd = new int[video_num];

	//����������㷨������Kruskal�㷨
	sort(matchInfoIn.begin(),matchInfoIn.end());  //����������operator �� ��������,ȨֵΪƥ������Ŀ
	int *setCount = new int[video_num];
	mLogIndex *mLIndex = new mLogIndex[video_num];
	for (int k=0;k<video_num;k++)
		setCount[k]=0;

	int setFlag=1;

	for(int k=(matchInfoIn.size()-1);k>=0;k--)
	{
		if((setCount[matchInfoIn[k].queryInx]==0)&&(setCount[matchInfoIn[k].trainInx]==0))
		{   
			setCount[matchInfoIn[k].queryInx]=setFlag;
			setCount[matchInfoIn[k].trainInx]=setFlag;
			setFlag++;
			mLIndex[k].inTree=true;  
			mLIndex[k].used = false;
		}
		else if ((setCount[matchInfoIn[k].queryInx]==0)&&(setCount[matchInfoIn[k].trainInx]!=0))
		{ 
			setCount[matchInfoIn[k].queryInx]=setCount[matchInfoIn[k].trainInx];
			mLIndex[k].inTree=true;  
			mLIndex[k].used = false;
		}
		else if ((setCount[matchInfoIn[k].queryInx]!=0)&&(setCount[matchInfoIn[k].trainInx]==0))
		{ 
			setCount[matchInfoIn[k].trainInx]=setCount[matchInfoIn[k].queryInx];
			mLIndex[k].inTree=true; 
			mLIndex[k].used = false;
		}
		else if (setCount[matchInfoIn[k].queryInx]!=setCount[matchInfoIn[k].trainInx])
		{   
			mLIndex[k].inTree=true; 
			mLIndex[k].used=false;
			int needChange=setCount[matchInfoIn[k].trainInx];
			int change=setCount[matchInfoIn[k].queryInx];
			for (int w=0;w<video_num;w++)
			{
				if (setCount[w]==needChange)
				{
					setCount[w]=change;
				}
			} //���¼��ϵĴ���
		} //������������������
		mLIndex[k].used=false;
	}
	delete []setCount;
	setCount = NULL;

	// Ѱ��ƥ�����Ŀ����ͼƬ
	for ( int i=0;i<video_num;i++)
	{
		imgInd[i]=0;
	}

	for (vector<mLog>::size_type i=0;i<matchInfoIn.size();i++)
	{
		imgInd[matchInfoIn[i].queryInx]+=(int)(matchInfoIn[i].matchPointIndex.size());
		imgInd[matchInfoIn[i].trainInx]+=(int)(matchInfoIn[i].matchPointIndex.size());
	}

	int baseImgFact=0;
	for (int i=0;i<video_num;i++)
	{
		if (imgInd[i]>imgInd[baseImgFact])
		{
			baseImgFact=i;
		}
	}

	__debug(
		cout << "[Info]Setting video input "<< baseImgFact <<" as base img..." << endl;
	)

	mtNode *head = new mtNode;
	head->fatherImgIdx = baseImgFact;
    head->nowImgIdx = baseImgFact;
	head->brother=NULL;
	videosFlag[baseImgFact]=true;

	delete [] imgInd;
	imgInd = NULL;

	__debug(
		cout << "[Info]Building match tree from match set..." << endl;
	)

	expandTree(matchInfoIn,mLIndex,head,videosFlag);
    __debug(printTree(head);)

	__debug(
		cout << "[Info]Adjusting match index and videos index..." << endl;
	)

    adjustVideos(imgSet,imgF,videosFlag);
	adjustIndx(matchInfoIn,head,videosFlag);
	return head;
}


/************************************************************************/
/*  �Ӹ��ڵ��������ƥ���ϵ�����õ��Ƿ������б�־����videosFlag */
/************************************************************************/
void parallaxWarper::expandTree(vector<mLog> &matchInfoIn, mLogIndex *mLIndex,
								mtNode *head,vector<bool>&videosFlag)
{
    int rootId=head->nowImgIdx;
	int sonNum=0;
	mtNode *present = NULL;
	//��ʼ����
	head->son=present;
	unsigned int matchSize=(unsigned int)matchInfoIn.size();

	for (unsigned int i=0;i<matchSize;i++)
	{
		//����ֻ�Դ��������ı߽��в���
		if ((mLIndex[i].inTree==true)&&(mLIndex[i].used==false))
		{
			if ((matchInfoIn[i].queryInx == rootId)||(matchInfoIn[i].trainInx == rootId))
			{
				if (matchInfoIn[i].queryInx == rootId)
				{   //���queryInx�Ǹ��ڵ㣬��ô�������matchIdx�Ķ���
					mLog matTemp;
					matTemp.trainInx = matchInfoIn[i].queryInx;
					matTemp.queryInx = matchInfoIn[i].trainInx;
					matTemp.matchPointIndex = matchInfoIn[i].matchPointIndexRev;
					matTemp.matchPointIndexRev = matchInfoIn[i].matchPointIndex;

					matchInfoIn[i]=matTemp;
				}
				if (sonNum==0)
				{
					present=new mtNode;
					head->son=present;
					sonNum++;
				}
				else
				{
					present->brother=new mtNode;
					present=present->brother;
					sonNum++;
				}

                present->fatherImgIdx=matchInfoIn[i].trainInx;
                present->nowImgIdx=matchInfoIn[i].queryInx;
				videosFlag[matchInfoIn[i].queryInx]=true;
				present->mLogIdx=i;
				present->brother=NULL;
				mLIndex[i].used=true;
			}
		}
	}

	present=head->son;
	while (present!=NULL)
	{
		expandTree(matchInfoIn,mLIndex,present,videosFlag);
		present=present->brother;
	}
	return;

}

/************************************************************************/
/* ɾ�������ͷ����ڴ�                                          */
/************************************************************************/
void parallaxWarper::releaseMatchTree(mtNode *head)
{
	cout <<"n:"<< head->fatherImgIdx<< endl;
	mtNode *present,*temp;
	present=head->son;
	while (present!=NULL)
	{
		temp=present->brother;
		releaseMatchTree(present);
		present=temp;
	}
    delete head;
	head = NULL;
}

/************************************************************************/
/* ��ӡƥ���ϵ��                                              */
/************************************************************************/
void parallaxWarper::printTree(mtNode *head)
{
	mtNode *present;

	cout <<"[Info]root "<< head->nowImgIdx<<":";

	present=head->son;
	while (present!=NULL)
	{
		cout << present->nowImgIdx<<"  ";
		present=present->brother;
	}
	cout << endl;
	present=head->son;
	while (present!=NULL){
		printTree(present);
		present=present->brother;
	}
}


/************************************************************************/
/*  ����videosFlag��videoSeq��imgFeature��ɾȥδ�����е�video��feature*/
/************************************************************************/
void parallaxWarper::adjustVideos(vector<Mat> &videosInOut, vector<imgFeatures> &imgFInOut, vector<bool> &videosFlag)
{
    vector<Mat> videosNew;
    vector<imgFeatures> imgFNew;

    for (unsigned int i=0;i<videosInOut.size();i++)
    {
        if (videosFlag[i]==true)//��������Ƶ������
        {
            videosNew.push_back(videosInOut[i]);
            imgFNew.push_back(imgFInOut[i]);
        }
    }

    videosInOut = videosNew;
    imgFInOut = imgFNew;
}


/************************************************************************/
/*  ����videosFlag����matchInfoIn��head�е�ͼ��������              */
/************************************************************************/
int parallaxWarper::adjustIndx(vector<mLog> &matchInfoIn, mtNode *head, vector<bool>&videosFlag)
{
	unsigned int videoSize = videosFlag.size();
	int *indxMap=new int[videoSize];

	int inx=0;
	for (unsigned int i=0;i<videoSize;i++)
	{
		if (videosFlag[i]==true)
		{  //��������Ƶ������
			indxMap[i]=inx;
			inx++;
		}else{
			indxMap[i]=-1;
		}
	}

	int ans=inx-1;

	for (unsigned int i=0;i<matchInfoIn.size();i++)
	{    //����mlog���������
		matchInfoIn[i].queryInx=indxMap[matchInfoIn[i].queryInx];
		matchInfoIn[i].trainInx=indxMap[matchInfoIn[i].trainInx];
		head->fatherImgIdx = indxMap[head->fatherImgIdx];
        head->nowImgIdx=indxMap[head->nowImgIdx];
	}

	delete [] indxMap;   //�ͷſռ�
	indxMap = NULL;
	return ans;
}



void parallaxWarper::mapForward(float x, float y, float &u, float &v, float *H_)
{
	u = H_[0]*x + H_[1]*y + H_[2];
	v = H_[3]*x + H_[4]*y + H_[5];
	float w =1/(H_[6]*x + H_[7]*y + 1);
	u = u*w;
	v = v*w;
}

void parallaxWarper::mapBackward(float u, float v, float &x, float &y, float *Hinv_)
{
	x = Hinv_[0]*u + Hinv_[1]*v + Hinv_[2];
	y = Hinv_[3]*u + Hinv_[4]*v + Hinv_[5];
	float z =1/(Hinv_[6]*u + Hinv_[7]*v + Hinv_[8]);
	x = x*z;
	y = y*z;
}

/************************************************************************/
/* �����������������ͶӰ����,�õ���ͶӰ��������Ϻ����¶���    */
/************************************************************************/
void parallaxWarper::detectGridResultRoi(Point &src_tl, Point &src_br, float *H_,
									     Point &dst_tl, Point &dst_br)
{

	float tl_x = static_cast<float>(src_tl.x);
	float tl_y = static_cast<float>(src_tl.y);
	float br_x = static_cast<float>(src_br.x);
	float br_y = static_cast<float>(src_br.y);

	float tl_uf = numeric_limits<float>::max();
	float tl_vf = numeric_limits<float>::max();
	float br_uf = -numeric_limits<float>::max();
	float br_vf = -numeric_limits<float>::max();

	float u, v;

    //�þ��ε��ĸ��˵���ȷ��dst ROI����С��Ӿ���
	mapForward(tl_x, tl_y, u, v, H_);
	tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
	br_uf = max(br_uf, u); br_vf = max(br_vf, v);

	mapForward(tl_x, br_y, u, v, H_);
	tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
	br_uf = max(br_uf, u); br_vf = max(br_vf, v);

	mapForward(br_x, tl_y, u, v, H_);
	tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
	br_uf = max(br_uf, u); br_vf = max(br_vf, v);

	mapForward(br_x, br_y, u, v, H_);
	tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
	br_uf = max(br_uf, u); br_vf = max(br_vf, v);

	dst_tl.x = static_cast<int>(tl_uf);
	dst_tl.y = static_cast<int>(tl_vf);
	dst_br.x = static_cast<int>(br_uf);
	dst_br.y = static_cast<int>(br_vf);
}


/************************************************************************/
/* ����ͼ���ĸ����������Թ���Ŀ��ͼ�����ϡ����¶���            */
/************************************************************************/
void parallaxWarper::detectResultRoiByCorner(Mat &srcImg, hMap &H, Point &dst_tl, Point &dst_br)
{
	float *Head = H.data;
	Point corner;

	float tl_uf = numeric_limits<float>::max();
	float tl_vf = numeric_limits<float>::max();
	float br_uf = -numeric_limits<float>::max();
	float br_vf = -numeric_limits<float>::max();

	float u, v;
	int nshift = 0;
	mapForward(0, 0, u, v, Head);
	tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
	br_uf = max(br_uf, u); br_vf = max(br_vf, v);

	nshift = H.width - 1;
	mapForward(0, static_cast<float>(srcImg.rows-1), u, v, Head+ 8*nshift);
	tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
	br_uf = max(br_uf, u); br_vf = max(br_vf, v);

	nshift = H.width*(H.height - 1);
	mapForward(static_cast<float> (srcImg.cols-1), 0, u, v, Head+ 8*nshift);
	tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
	br_uf = max(br_uf, u); br_vf = max(br_vf, v);

	nshift = H.width*H.height - 1;
	mapForward(static_cast<float>(srcImg.cols-1),static_cast<float>(srcImg.rows-1), u, v, Head+ 8*nshift);
	tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
	br_uf = max(br_uf, u); br_vf = max(br_vf, v);

	dst_tl.x = static_cast<int>(tl_uf);
	dst_tl.y = static_cast<int>(tl_vf);
	dst_br.x = static_cast<int>(br_uf);
	dst_br.y = static_cast<int>(br_vf);
}




/************************************************************************/
/* ����ͼ��߽��ͶӰ����õ�Ŀ��ͼ������ϡ����¶���           */
/************************************************************************/
void parallaxWarper::detectResultRoiByBother(Mat &srcImg, hMap &H, Point &dst_tl, Point &dst_br)
{

	float *Head = H.data;
	Point tmp_tl,tmp_br;
	float u, v;
	int nshift=0;
	int width = srcImg.cols;
	int height = srcImg.rows;
	int rows = H.height;
	int cols = H.width;

	float tl_uf = numeric_limits<float>::max();
	float tl_vf = numeric_limits<float>::max();
	float br_uf = -numeric_limits<float>::max();
	float br_vf = -numeric_limits<float>::max();

	for(int i=0;i<width;i++)
	{
		nshift = i/GridSize;//��GridSizeΪ2��ָ��������˴�������λ���� 
		mapForward(static_cast<float>(i), 0, u, v, Head + 8*nshift);
		tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
		br_uf = max(br_uf, u); br_vf = max(br_vf, v);

		nshift = cols*(rows-1) + nshift;
		mapForward(static_cast<float>(i), static_cast<float>(height-1), u, v, Head + 8*nshift);
		tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
		br_uf = max(br_uf, u); br_vf = max(br_vf, v);

	}

	for(int j=0;j<height;j++)
	{
		nshift = (j/GridSize)*cols;
		mapForward(0, static_cast<float>(j), u, v, Head + 8*nshift);
		tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
		br_uf = max(br_uf, u); br_vf = max(br_vf, v);

		nshift = nshift + cols -1;
		mapForward(static_cast<float> (width-1), static_cast<float>(j), u, v, Head + 8*nshift);
		tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
		br_uf = max(br_uf, u); br_vf = max(br_vf, v);
	}

	dst_tl.x = static_cast<int>(tl_uf);
	dst_tl.y = static_cast<int>(tl_vf);
	dst_br.x = static_cast<int>(br_uf);
	dst_br.y = static_cast<int>(br_vf);

}

void drawline(Point2f from, Point2f to, Mat &image)
{

	Point f, t;
	f.x = static_cast<int>(from.x);
	f.y = static_cast<int>(from.y);
	t.x = static_cast<int>(to.x);
	t.y = static_cast<int>(to.y);

	circle(image, t, 1,Scalar(255),1);

	if(abs(f.x-t.x)<=1&&abs(f.y-t.y)<=1)//���㱾��������ͨ��ʱ�򲻻���
	{
		return;
	}

	line(image, f, t, Scalar(255),3);
}


void parallaxWarper::detectMask(Mat &srcImg, hMap &H, Point topleft, Mat &mask)
{
	CV_Assert(mask.type()==CV_8UC1);

	float *Head = H.data;
	Point tmp_tl,tmp_br;
	float u, v;
	int nshift=0;
	int width = srcImg.cols;
	int height = srcImg.rows;
	int rows = H.height;
	int cols = H.width;

	Point2f pointEdge1;
	Point2f pointEdge2;
	Point2f pointEdge3;
	Point2f pointEdge4;

  /*�����ʵ������Ҫ�棬���ڱ�1��4����㣬��1β���Ǳ�2��㣬��2�ͱ�3��β�㣬��3������Ǳ�4��β��*/
  /*�����������Ѿ������ӵģ�����Ҫ�ر���ȥ���ӵ�,���Գ�ʼ��Ҳ���ô���*/

  /*Point2f pointCorner1;
	Point2f pointCorner2;
	Point2f pointCorner3;
	Point2f pointCorner4;*/

	mapForward(0, 0, u, v, Head);
	pointEdge1.x = u-topleft.x; pointEdge1.y = v-topleft.y;//��ʼ��
	//pointCorner1.x = u; pointCorner1.y = v;//�涥�㣬��ͬ
	
	pointEdge4.x = u-topleft.x; pointEdge4.y = v-topleft.y;
	//pointCorner4.x = u; pointCorner4.y = v;

	nshift = H.width - 1;
	mapForward(0, static_cast<float>(srcImg.rows-1), u, v, Head+ 8*nshift);
	pointEdge3.x = u-topleft.x; pointEdge3.y = v-topleft.y;
	//pointCorner2.x = u; pointCorner2.y = v;

	nshift = H.width*(H.height - 1);
	mapForward(static_cast<float> (srcImg.cols-1), 0, u, v, Head+ 8*nshift);
	pointEdge2.x = u-topleft.x; pointEdge2.y = v-topleft.y;
	//pointCorner3.x = u; pointCorner3.y = v;

	//�ҵ�����ͶӰͼ�����СROI����mask�ϻ��������߽�
	Point2f pointTmp;
	for(int i=0;i<width;i++)
	{
		nshift = i/GridSize;//��GridSizeΪ2��ָ��������˴�������λ���� 
		mapForward(static_cast<float>(i), 0, u, v, Head + 8*nshift);//��1
		pointTmp.x = u-topleft.x; pointTmp.y = v-topleft.y;
		drawline(pointEdge1, pointTmp, mask);
		pointEdge1 = pointTmp;

		nshift = cols*(rows-1) + nshift;
		mapForward(static_cast<float>(i), static_cast<float>(height-1), u, v, Head + 8*nshift);//��3
		pointTmp.x = u-topleft.x; pointTmp.y = v-topleft.y;
		drawline(pointEdge3, pointTmp, mask);
		pointEdge3 = pointTmp;
	}


	for(int j=0;j<height;j++)
	{
		nshift = (j/GridSize)*cols;
		mapForward(0, static_cast<float>(j), u, v, Head + 8*nshift);//��4
		pointTmp.x = u-topleft.x; pointTmp.y = v-topleft.y;
		drawline(pointEdge4, pointTmp, mask);
		pointEdge4 = pointTmp;

		nshift = nshift + cols -1;
		mapForward(static_cast<float> (width-1), static_cast<float>(j), u, v, Head + 8*nshift);//��2
		pointTmp.x = u-topleft.x; pointTmp.y = v-topleft.y;
		drawline(pointEdge2, pointTmp, mask);
		pointEdge2 = pointTmp;
	}

	/*
	//����������
	drawline(pointEdge1, pointCorner2, mask);
	drawline(pointEdge2, pointEdge3, mask);
	drawline(pointCorner3, pointEdge4, mask);
	drawline(pointCorner4, pintCorner1, mask);*/


	Point seed;
	seed.x = static_cast<int>(0.5*(pointEdge1.x+pointEdge4.x));
	seed.y = static_cast<int>(0.5*(pointEdge1.y+pointEdge4.y));

	cout<<"seed: "<<seed.x<<", "<<seed.y<<endl;
	cout<<"mask size:"<<mask.size()<<endl;
	cout<<"topleft:"<<topleft.x<<" "<<topleft.y<<endl;

	Rect DontKnowWhy;
	floodFill(mask, seed, Scalar(255),&DontKnowWhy,Scalar(20, 20, 20),Scalar(20, 20, 20),FLOODFILL_FIXED_RANGE);

	cvShowImage("Mask", &IplImage(mask));
	cvWaitKey(0);
	
}

//���� 2014/7/23 9:54:13
float avgMask(Mat &map, int x, int y)
{
    float sum=0.f;
    int count=0;

    for(int i = -1;i<2; i++)
        for(int j = -1;j<2; j++)
        {
            if(map.at<float>(x+i,y+j)!=0.f)
            {
                sum = sum + map.at<float>(x+i,y+j);
                count++;
            }
        }

    if(sum!=0.f)
        return sum/count;

    return map.at<float>(x,y);
}


/************************************************************************/
/* ����ӳ���xmap, ymap                                                */
/************************************************************************/
Rect parallaxWarper::buildMaps(Mat &srcImg, hMap &H, Mat &xmap, Mat &ymap, Mat &mask)
{
	Point dst_tl, dst_br;

	detectResultRoiByBother(srcImg, H, dst_tl, dst_br);//��ȡ�Ա߽�׼ȷ����ROI��ʽ
	//detectResultRoiByCorner(srcImg, H, dst_tl,dst_br);//��ȡ�Խǵ�ֹ���ROI��ʽ

	int doi_rows = dst_br.y - dst_tl.y;
	int doi_cols = dst_br.x - dst_tl.x;

	xmap.create(doi_rows + 1, doi_cols + 1, CV_32F);
	ymap.create(doi_rows + 1, doi_cols + 1, CV_32F);
	mask.create(doi_rows + 1, doi_cols + 1, CV_8UC1);

	xmap.setTo(Scalar::all(0.f));
	ymap.setTo(Scalar::all(0.f));
	

	if (dst_tl.x==0 &&dst_tl.y==0)//����ǲο�ͼ����ֱ������mask
	{
		mask.setTo(Scalar::all(255));
	} 
	else//���ǲο�ͼ����ͨ���߼����mask
	{
		mask.setTo(Scalar::all(0));
		detectMask(srcImg, H, dst_tl, mask);
	}


	int nCols = srcImg.cols;
	int nRows = srcImg.rows;
	float *head = H.data;
	//float *present = H.data;
	int nshift;
	float u, v;
	int nIdx_x, nIdx_y;
	int sum1 = 0; int sum2 = 0;
	int count=0;
	int tl_x=0;int tl_y=0;
	

	//��������������е�ǰͶ���ع�ʹ��ÿ���ڵ�����֮����һ�����غϣ�
	for(int j=0; j<H.height; j++)
	{
		for(int i = 0; i<H.width; i++)
		{
			nshift = j*(H.width)+i;
			for(int x = 0; x<GridSize+4; x++)//��֤�����������غ�
				for(int y = 0; y<GridSize+4; y++)
			     {

				    mapForward(static_cast<float>(tl_x + x), static_cast<float>(tl_y + y), u, v, head + 8*nshift);

					if ( v<dst_tl.y||u<dst_tl.x)
					{
						sum1++;//��һ��������(����dst_roi�߽�)
						continue;
					}
					if((v-dst_tl.y>doi_rows)||(u-dst_tl.x>doi_cols))
					{
						sum1++;
						continue;
					}

					nIdx_x = static_cast<int> (u - dst_tl.x);
					nIdx_y = static_cast<int> (v - dst_tl.y);

					if(x<nRows&&y<nCols)
					{
						xmap.at<float>(nIdx_y, nIdx_x) = static_cast<float>(x+tl_x);				
						ymap.at<float>(nIdx_y, nIdx_x) = static_cast<float>(y+tl_y);
						count++;
					}
			    }
           
				tl_x = i*GridSize;
		}
		tl_y = j*GridSize;
	}
	/*
    //��ͼ�����ظ��������е�ǰͶ
	for(int y=0;y<nRows;y++)
	{
		nshift = y/GridSize;

		present = H.data + nshift*H.width*8;
		
		for(int x=0;x<nCols;x++)
		{
			nshift = x/GridSize;
			count++;

			mapForward(static_cast<float>(x), static_cast<float>(y), u, v, present + nshift*8 );
			
			if ( v<dst_tl.y||u<dst_tl.x)
			{
				sum1++;//��һ��������(����dst_roi�߽�)
				continue;
			}
			if((v-dst_tl.y>doi_rows)||(u-dst_tl.x>doi_cols))
			{
				sum1++;
				continue;
			}

			nIdx_x = static_cast<int> (u - dst_tl.x);
			nIdx_y = static_cast<int> (v - dst_tl.y);

			xmap.at<float>(nIdx_y, nIdx_x) = static_cast<float>(x);
			ymap.at<float>(nIdx_y, nIdx_x) = static_cast<float>(y);
		}
	}*/

	//���
	for(int k = 1; k<xmap.cols-1;k++)
		for(int h=1; h<xmap.rows-1;h++)
		{
			/* //������
			if ((xmap.at<float>(h,k)==0.f)&&(ymap.at<float>(h,k)==0.f))
			{
				xmap.at<float>(h,k)=xmap.at<float>(h,k-1);
				ymap.at<float>(h,k)=ymap.at<float>(h,k-1);	
				sum2++;//�ڶ���������(���Ŀն�)	
			}
			*/
			if ((xmap.at<float>(h,k)==0.f)&&(ymap.at<float>(h,k)==0.f))
			{
				xmap.at<float>(h,k) = avgMask(xmap, h, k);//3*3��Ĥƽ����
				ymap.at<float>(h,k) = avgMask(ymap, h, k);
			}
		}


	cout<<"sum1 = "<<sum1<<", sum2 ="<<sum2<<endl;
	
	return Rect(dst_tl, dst_br);
}










