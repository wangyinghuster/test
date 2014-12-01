#ifndef FRAMEWARPER_H
#define FRAMEWARPER_H

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

#include "featuresMatcher.h"
#include "KRwarper.h"
#include "KRestimator.h"
#include "macrosConfig.h" //‘§±‡“Î∫Í≈‰÷√Œƒº˛

using namespace std;
using namespace cv;

class frameWarper
{
public:
    frameWarper();
	virtual ~frameWarper(){}

    virtual void prepare(vector<Mat> &imgSet,vector<imgFeatures> &imgF,vector<mLog> &matchInfoIn,
						 vector<Mat> &imgMaskWarpOut,vector<Point> &topleftOut,vector<bool> &videoFlag)=0;

    virtual void doWarp(vector<Mat> &imgSet,vector<Mat> &imgWarpOut)=0;

};


template<class W> 
class KRBasedWarp:public frameWarper
{
public:
	~KRBasedWarp(void){}

	void prepare(vector<Mat> &imgSet, vector<imgFeatures>&imgF,vector<mLog> &matchInfoIn, 
			     vector<Mat> &imgMaskWarpOut, vector<Point> &topleftOut,vector<bool> &videoFlag);
	void doWarp(vector<Mat> &imgSet,vector<Mat>&imgWarpOut);

public:

	void estimate(vector<Mat> &imgSet, vector<imgFeatures> &imgFeature,vector<mLog> &matchInfoIn, 
		          vector<CameraParams> &cameras);
	Rect prepare(const Mat &src, CameraParams &camera, Mat &xmap, Mat &ymap);

	void warp(const Mat &src, Mat &xmap, Mat &ymap, int interp_mode, int border_mode,Mat &dst);


protected:
	float scale;
	W warper;
	KREstimator kr_estimator;
	vector<Mat> xmap_list;
	vector<Mat> ymap_list;
};


class PlaneKRWarper:public KRBasedWarp<PlaneWarper>
{
public:
	PlaneKRWarper(float scale_ = 1.f){ scale = scale_;}
	~PlaneKRWarper(void){}
};

class CylindricalKRWarper:public KRBasedWarp<CylindricalWarper>
{
public:
	CylindricalKRWarper(float scale_ = 1.f){ scale = scale_;}
	~CylindricalKRWarper(void){}
};

class SphericalKRWarper:public KRBasedWarp<SphericalWarper>
{
public:
	SphericalKRWarper(float scale_ = 1.f){ scale = scale_; }
	~SphericalKRWarper(void){}
};


#endif // FRAMEWARPER_H
