#ifndef OVERLAPSEAMFIND_H
#define OVERLAPSEAMFIND_H

#include "seamFinder.h"

typedef struct overlapLog{    //重叠区对应的轮廓信息
    int highInx;   //轮廓所在的图像索引(在匹配树高层级)
    int lowInx;   //轮廓所在的图像索引(在匹配树低层级)
    int value;  //轮廓的值
} oLog;

class overlapSeamFinder: public seamFinder//找到warpImage对应的重叠区的轮廓，存放在seamMsk里
{
public:
	virtual void findSeam(vector<Mat> &warpImages, vector<Mat> &warpMask, vector<Mat> &seamMsk, vector<Point> &topleft);
	
private:
	void  process(
            const Mat &warpMask1, int index1, const Mat &warpMask2, int index2,Point tl1, Point tl2, Mat &mask1, Mat &mask2);
	void find(const std::vector<Mat> &warpMasks, const std::vector<Point> &corners,
                      std::vector<Mat> &seamMasks);
	class ImagePairLess
    {
    public:
        ImagePairLess(const std::vector<Mat> &images, const std::vector<Point> &corners)
            : src_(&images[0]), corners_(&corners[0]) {}

        bool operator() (const std::pair<size_t, size_t> &l, const std::pair<size_t, size_t> &r) const
        {
            Point c1 = corners_[l.first] + Point(src_[l.first].cols / 2, src_[l.first].rows / 2);
            Point c2 = corners_[l.second] + Point(src_[l.second].cols / 2, src_[l.second].rows / 2);
            int d1 = (c1 - c2).dot(c1 - c2);

            c1 = corners_[r.first] + Point(src_[r.first].cols / 2, src_[r.first].rows / 2);
            c2 = corners_[r.second] + Point(src_[r.second].cols / 2, src_[r.second].rows / 2);
            int d2 = (c1 - c2).dot(c1 - c2);

            return d1 < d2;
        }

    private:
        const Mat *src_;
        const Point *corners_;
    };
	 // processing images pair data
    Point unionTl_, unionBr_;
    Size unionSize_;
    Mat_<uchar> mask1_, mask2_;
    Mat_<uchar> contour1mask_, contour2mask_;
};




#endif