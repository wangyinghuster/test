#ifndef KR_ESTIMATOR_HPP
#define KR_ESTIMATOR_HPP

/*LSH's alteration based on OpenCV 2.4.6*/
#include "featuresMatcher.h"
#include "featuresGenerator.h"
#include "sysException.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"

#include <string>
#include <vector>
#include <iostream>

#ifndef  __NOT_USE_OCL_LIB

# include "opencv2/gpu/gpu.hpp"
#endif

using namespace std;
using namespace cv;
using namespace cv::detail;

static const float conf_thrd = 0.5;

typedef struct homoTree{    //匹配树关系图
	int picInx;             //本节点的图片INDEX
	int matId;              //关联父子节点图像的匹配的索引号。
	struct homoTree* next;  //本节点的兄弟节点
	struct homoTree* son;   //本节点的子节点
} homoT;


struct exmLog: matchLog
{
    bool used;     //查看这条匹配关系是否被用过
    bool inTree;   //查看这条匹配关系是否在树中

    Mat H;         //正变换的单应性矩阵 query到train
    Mat HRev;      //逆变换的单应性矩阵，train到query

    vector<uchar> inliers_mask;    // 内点mask  
    int num_inliers;               // 内点数

    vector<uchar> inliers_mask_rev;
    int num_inliers_rev;               

};


class OptimizeMatch
{
public:
	OptimizeMatch(){}
	~OptimizeMatch(){}

	int expandMatchInfo(vector<Mat> &imgSet, vector<imgFeatures> &imgF,vector<mLog> &matchIdx,homoT *root);
	
	int calAllMatch(vector<Mat> &imgSet, vector<imgFeatures> &imgF,vector<mLog> &matchIdx);
	int adjustIndx(homoT *root, unsigned int videoSize);
	int updateTreIdx(homoT *root, int idxMap[], int matIdxMap[]);
	int adjustVideos(vector<Mat> &videoAdjust);
	
	// Max spanning Tree algorithm
	int printTree(homoT *root);
	int deleteTree(homoT *root);
	int buildHomoTree(homoT *root);
	void MaxSpanningTree(homoT *homoTRoot,unsigned int video_num);

public:
	vector<bool> videosFlag;
	vector<exmLog> match_info;

};


/*parameter estimation part*/

class KREstimator
{

public:
	KREstimator(void){ myBundleAdjusterRay_(4, 3);}
	~KREstimator(void){}

	bool estimate(vector<Mat> &imgSet,vector<imgFeatures>&imgF, vector<mLog> &match_info,
				  vector<CameraParams> &cameras);

private:

	void estimatefromHomography( vector<Mat> &imgSet, vector<exmLog> &match_info, homoT *root, 
								 vector<CameraParams> &cameras);

	//using inliers to adjust camera parameters
	void estimateBunderadjuster(vector<Mat> &imgSet, vector<imgFeatures> &imgF, vector<exmLog> &match_info,
								homoT *root, vector<CameraParams> &cameras);

	void calcRotation( vector<exmLog>&match_info, homoT *root, vector<CameraParams> &cameras);

	void estimateFocal( vector<Mat> &imgSet,vector<exmLog> &match_info, vector<double>&focals);

	// function for bunderadjuster
	CvTermCriteria termCriteria() { return term_criteria_; }
	void setTermCriteria(const CvTermCriteria& term_criteria) { term_criteria_ = term_criteria; }


private:

	void setUpInitialCameraParams(const std::vector<CameraParams> &cameras);
	void obtainRefinedCameraParams(std::vector<CameraParams> &cameras) const;
	void calcError( vector<Mat> &imgSet, vector<imgFeatures> &imgF, vector<exmLog> &match_info, Mat &err);
	void calcJacobian( vector<Mat>&imgset, vector<imgFeatures>&imgF,vector<exmLog> &match_info, Mat &jac);

	void myBundleAdjusterRay_(int num_params_per_cam, int num_errs_per_measurement)
	{
		num_params_per_cam_ = num_params_per_cam;
		num_errs_per_measurement_ = num_errs_per_measurement;
		setTermCriteria(cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 1000, DBL_EPSILON));
	}

public:
	OptimizeMatch matcher;
private:
	Mat err1_, err2_;
	int num_images_;
	int total_num_matches_;

	int num_params_per_cam_;
	int num_errs_per_measurement_;

	CvTermCriteria term_criteria_; //Levenberg–Marquardt algorithm termination criteria

	Mat cam_params_; // Camera parameters matrix (CV_64F)

};



#endif