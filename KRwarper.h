#ifndef KRWARPBASE_HPP
#define KRWARPBASE_HPP

/*LSH's alteration based on OpenCV 2.4.6*/

#include "sysException.h"
#include "macrosConfig.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv_modules.hpp"


#include <string>
#include <vector>
#include <iostream>

#ifndef  __NOT_USE_OCL_LIB

# include "opencv2/gpu/gpu.hpp"
#endif

using namespace std;
using namespace cv;
//using namespace cv::detail;



struct CV_EXPORTS WarperProjector
{
	void setCameraParams(const Mat &K = Mat::eye(3, 3, CV_32F),
		const Mat &R = Mat::eye(3, 3, CV_32F),
		const Mat &T = Mat::zeros(3, 1, CV_32F));

	float scale;
	float k[9];
	float rinv[9];
	float r_kinv[9];
	float k_rinv[9];
	float t[3];

};

struct CV_EXPORTS CylindricalProjector:WarperProjector
{
	void mapForward(float x, float y, float &u, float &v);
	void mapBackward(float u, float v, float &x, float &y);
};

struct CV_EXPORTS PlaneProjector:WarperProjector
{
	void mapForward(float x, float y, float &u, float &v);
	void mapBackward(float u, float v, float &x, float &y);
};

struct CV_EXPORTS SphericalProjector:WarperProjector
{
	void mapForward(float x, float y, float &u, float &v);
	void mapBackward(float u, float v, float &x, float &y);
};
 

class KRWarper
{
public:
	virtual ~KRWarper(){}

	virtual Point2f warpPoint(const Point2f &pt, const Mat &K, const Mat &R)=0;

	virtual Rect buildMaps(Size src_size, const Mat &K, const Mat &R, Mat &xmap, Mat &ymap)=0;

	virtual Point warp( const Mat &src, const Mat &K, const Mat &R, int interp_mode, 
	int border_mode, Mat &dst )=0;

	virtual Rect warpRoi(Size src_size, const Mat &K, const Mat &R)=0;

	virtual Rect prepare(const Mat &src, const Mat &K, const Mat &R, Mat &xmap, Mat &ymap)=0;

	virtual void doWarp(const Mat &src, Mat &xmap, Mat &ymap, int interp_mode, int border_mode, Mat &dst)=0;

	void setScale(float){}

};


template <class P>
class CV_EXPORTS KRWarperbase : public KRWarper
{
public:
	Point2f warpPoint(const Point2f &pt, const Mat &K, const Mat &R);
	Rect buildMaps(Size src_size, const Mat &K, const Mat &R, Mat &xmap, Mat &ymap);
	Point warp(const Mat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode, Mat &dst);

	Rect prepare( const Mat &src, const Mat &K, const Mat &R, Mat &xmap, Mat &ymap);
	void doWarp( const Mat &src, Mat &xmap, Mat &ymap, int interp_mode, int border_mode, Mat &dst);

	void warpBackward(const Mat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
		Size dst_size, Mat &dst);

	Rect warpRoi(Size src_size, const Mat &K, const Mat &R);

	float getScale() const { return projector_.scale; }
	void setScale( float scale) { projector_.scale = scale;   }

protected:

	// Detects ROI of the destination image. It's correct for any projection.
	virtual void detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br);

	// Correctness for any projection isn't guaranteed.
	void detectResultRoiByBorder(Size src_size, Point &dst_tl, Point &dst_br);

	P projector_;
};


class CylindricalWarper:public KRWarperbase<CylindricalProjector>
{
public:
	CylindricalWarper(float scale = 1.f) { projector_.scale = scale; }

protected:
	void detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br)
	{
		KRWarperbase<CylindricalProjector>::detectResultRoiByBorder(src_size, dst_tl, dst_br);
	}

};


class SphericalWarper:public KRWarperbase<SphericalProjector>
{
public:
	SphericalWarper(float scale = 1.f) { projector_.scale = scale; }

protected:

	void detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br);
};


class PlaneWarper:public KRWarperbase<PlaneProjector>
{
public:
	PlaneWarper(float scale = 1.f) { projector_.scale = scale; }

protected:
	void detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br);

};


#endif//End for KRWARPBASE_HPP