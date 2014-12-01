#ifndef PARALLEXWARPER_H
#define PARALLEXWARPER_H

#include "frameWarper.h"
#include "sysException.h"

using namespace cv;
using namespace std;

struct mLogIndex
{
	bool inTree;
	bool used;
};

typedef struct matchTreeNode
{
    int mLogIdx;  //对应的mLog应该满足这个条件：mlog中的trainIdx==fatherImgIdx
    int fatherImgIdx;
	int nowImgIdx;
    struct matchTreeNode* brother;
    struct matchTreeNode* son;
    matchTreeNode(int mIdx, int faImgIdx ,int nIdx)
        :mLogIdx(mIdx), fatherImgIdx(faImgIdx), nowImgIdx(nIdx), brother(NULL), son(NULL){}
	matchTreeNode()
        :mLogIdx(0), fatherImgIdx(0), nowImgIdx(0), brother(NULL), son(NULL){}
	
} mtNode;


typedef struct homographyMatrixMap
{
	static const int paraNum=8;
	int height;
	int width;
	float *data;
	homographyMatrixMap(int heightIn,int widthIn)
		:data(NULL),height(heightIn),width(widthIn)
	{
		data=new float[height*width*paraNum];
		if (data==NULL) throw sysException("Unable to allocate hMap data,insufficient memory!");
	}

	homographyMatrixMap()
		:data(NULL),height(0),width(0){}

	~homographyMatrixMap(){
		if (data!=NULL) delete data;
	}

	void creat(int heightIn,int widthIn){
		if (data!=NULL) {delete data;data=NULL;}
		height= heightIn;
		width= widthIn;
		data=new float[height*width*paraNum];
		if (data==NULL) throw sysException("Unable to allocate hMap data,insufficient memory!");
	}

    MatExpr at(int x,int y){   //这个返回需要用MatExpr类型
        if ((x>=width)||(y>=height)) throw sysException("Hmap access violate!");
        Mat H;
        float *p=data+(y*width+x)*paraNum;
        H.create(3,3,CV_32FC1);
        H.at<float>(0,0)=p[0];  //这里存入8个参数，根据hMap::paraNum的需要修改
        H.at<float>(0,1)=p[1];
        H.at<float>(0,2)=p[2];
        H.at<float>(1,0)=p[3];
        H.at<float>(1,1)=p[4];
        H.at<float>(1,2)=p[5];
        H.at<float>(2,0)=p[6];
        H.at<float>(2,1)=p[7];
        H.at<float>(2,2)=1;
        return MatExpr(H);
    }

    void modify(int x,int y,Mat &H){
        if (H.type()!=CV_32FC1) throw sysException("hMap needs H of CV_32FC1!");
        if ((x>=width)||(y>=height)) throw sysException("Hmap modifying violate!");
        float *p=data+(y*width+x)*paraNum;
        float s=H.at<float>(2,2);
        p[0]=H.at<float>(0,0)/s;  //这里保存8个参数，根据hMap::paraNum的需要修改
        p[1]=H.at<float>(0,1)/s;
        p[2]=H.at<float>(0,2)/s;
        p[3]=H.at<float>(1,0)/s;
        p[4]=H.at<float>(1,1)/s;
        p[5]=H.at<float>(1,2)/s;
        p[6]=H.at<float>(2,0)/s;
        p[7]=H.at<float>(2,1)/s;
    }

} hMap;


class parallaxWarper : public frameWarper
{
public:
    parallaxWarper();

    void prepare(vector<Mat> &imgSet, vector<imgFeatures> &imgF, vector<mLog> &matchInfoIn,
                 vector<Mat> &imgMaskWarpOut, vector<Point> &topleftOut, vector<bool> &videoFlag);
    void doWarp(vector<Mat> &imgSet, vector<Mat> &imgWarpOut);

private:

   // static double QualityThreshold;  //质量评估的阈值
   //static double randomSeedHThreshold;  //添加点的时候与h相似度接近的阈值

   // void estimateQuality(Mat &H,Mat &Hs,bool &isAceept);

   /* void calculateLevel(vector<imgFeatures> &imgF,vector<mLog> matchInfoIn,
                         vector<vector<int>> &keyPointIdxOut,vector<Mat> &HSets);*/
   
	//最大匹配树生成,裁剪imgset,修改matchInfoIn

    mtNode* buildMatchTree( vector<Mat>&imgSet, vector<imgFeatures> &imgF,vector<mLog> &matchInfoIn, vector<bool> &videosFlag);
   
	void releaseMatchTree(mtNode *head);

    void calculateLocalH( vector<Mat>&imgSet, vector<imgFeatures> &imgF, vector<mLog> &matchInfoIn,
		                  mtNode *head,vector<hMap> &hmap_list_out);

	void preWarp( vector<Mat>&imgSet, vector<hMap> &hmap_list, vector<Mat>& imgMasks,
		          vector<Point> &topleftOut);

private:

    void findHSet(imgFeatures &imgFsrc,imgFeatures &imgFdst,
                  vector<DMatch> &matchIn,
                  vector<vector<int>> &keyPointIdxOut,vector<pair<Mat,Mat>> &HSetsOut);
    bool isCross(Point2f s1,Point2f dirct1,Point2f d1,Point2f d2); //判断射线和向量是否相交

	//匹配树生成的子函数
	void expandTree( vector<mLog> &matchInfoIn, mLogIndex *mLIndex, mtNode *head, 
		             vector<bool> &videoFlag);
	void printTree( mtNode *head );
    void adjustVideos(vector<Mat> &videosInOut, vector<imgFeatures> &imgFInOut,vector<bool> &videosFlag);
	int adjustIndx(vector<mLog> &matchInfoIn, mtNode *head,vector<bool>&videosFlag);

	//warp子函数
	void mapForward(float x, float y, float &u, float &v, float *H_);
	void mapBackward(float u, float v, float &x, float &y, float *Hinv_);
	void detectGridResultRoi(Point &src_tl, Point &src_br, float *H_,Point &dst_tl, Point &dst_br);
	void detectResultRoiByCorner(Mat &srcImg, hMap &H, Point &dst_tl, Point &dst_br);
	void detectResultRoiByBother(Mat &srcImg, hMap &H, Point &dst_tl, Point &dst_br);
	void detectMask(Mat &srcImg, hMap &H, Point topleft, Mat &mask);
	Rect buildMaps(Mat &srcImg,  hMap &H, Mat &xmap, Mat &ymap, Mat& mask);

private:
	vector<Mat> xmap_list;
	vector<Mat> ymap_list;
    static const int GridSize = 16;
};

#endif // PARALLEXWARPER_H
