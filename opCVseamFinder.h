#ifndef OPCVSEAMFIND_H
#define OPCVSEAMFIND_H

#include "seamFinder.h"

using namespace std;
using namespace cv;
using namespace detail;

class graphCutSeamFinder:public seamFinder
{
public:
    graphCutSeamFinder(bool grad=true):useGrad(true){useGrad=grad;}
    void findSeam(vector<Mat> &warpImages,vector<Mat> &warpMask,vector<Mat> &seamMsk,vector<Point> &topleft);

private:
    bool useGrad;
};

class DPSeamFinder:public seamFinder
{
public:
    DPSeamFinder(bool grad=true):useGrad(true){useGrad=grad;}
    void findSeam(vector<Mat> &warpImages,vector<Mat> &warpMask,vector<Mat> &seamMsk,vector<Point> &topleft);

private:
    bool useGrad;
};

#endif // OPCVSEAMFIND_H
