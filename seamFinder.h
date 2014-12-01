#ifndef SEAMFINDER_H
#define SEAMFINDER_H


#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "macrosConfig.h" //Ԥ����������ļ�

using namespace std;
using namespace cv;

class seamFinder
{
public:
    seamFinder();
    virtual ~seamFinder()=0;

	virtual void findSeam(vector<Mat> &warpImages,vector<Mat> &warpMask,vector<Mat> &seamMsk,vector<Point> &topleft)=0;
	virtual void adjust(Mat &foreWarpMask){};

};

class  noSeamFinder : public seamFinder   //����һ������class
{
public:
    void findSeam(vector<Mat> & , vector<Mat> & , vector<Mat> & , vector<Point> & ) {}
};

#endif // SEAMFINDER_H
