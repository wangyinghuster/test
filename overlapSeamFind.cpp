#include "overlapSeamFind.h"


void  overlapSeamFinder::findSeam(vector<Mat> &warpImages, vector<Mat> &warpMask, vector<Mat> &seamMsk, vector<Point> &topleft)
{
	seamMsk.clear();
    for (unsigned int i=0;i<warpMask.size();i++){
		seamMsk.push_back(Mat::zeros(warpMask[i].size(),warpMask[i].type()));   //这里需要这么复制，否则seamMask就和imgMask共用一块内存区域。

    }
    find(warpMask,topleft,seamMsk);

}

void overlapSeamFinder::find(const std::vector<Mat> &warpMasks, const std::vector<Point> &corners,
                      std::vector<Mat> &masks)
{
    vector<pair<size_t, size_t> > pairs;

    for (size_t i = 0; i+1 < warpMasks.size(); ++i)
        for (size_t j = i+1; j < warpMasks.size(); ++j)
            pairs.push_back(make_pair(i, j));

    sort(pairs.begin(), pairs.end(), ImagePairLess(warpMasks, corners));   //按照每对之间的中心点的距离进行排序
    reverse(pairs.begin(), pairs.end());  //逆序过来，也就是距离从远到近

    for (size_t i = 0; i < pairs.size(); ++i)
    {
        size_t i0 = pairs[i].first, i1 = pairs[i].second;
        process(warpMasks[i0], i0,warpMasks[i1], i1,corners[i0], corners[i1], masks[i0], masks[i1]);
    }
}


void overlapSeamFinder::process(
            const Mat &warpMask1, int index1, const Mat &warpMask2, int index2,Point tl1, Point tl2, Mat &mask1, Mat &mask2)
{
    CV_Assert(warpMask1.size() == mask1.size());
    CV_Assert(warpMask2.size() == mask2.size());

    Point intersectTl(std::max(tl1.x, tl2.x), std::max(tl1.y, tl2.y));

    Point intersectBr(std::min(tl1.x + warpMask1.cols, tl2.x + warpMask2.cols),
                      std::min(tl1.y + warpMask1.rows, tl2.y + warpMask2.rows));

    if (intersectTl.x >= intersectBr.x || intersectTl.y >= intersectBr.y)
        return; // there are no conflicts //计算重叠区域，没有相交

    unionTl_ = Point(std::min(tl1.x, tl2.x), std::min(tl1.y, tl2.y));

    unionBr_ = Point(std::max(tl1.x + warpMask1.cols, tl2.x + warpMask2.cols),
                     std::max(tl1.y + warpMask1.rows, tl2.y + warpMask2.rows));

    unionSize_ = Size(unionBr_.x - unionTl_.x, unionBr_.y - unionTl_.y);  //若有相交，则计算的是合并之后的大小

    mask1_ = Mat::zeros(unionSize_, CV_8U);
    mask2_ = Mat::zeros(unionSize_, CV_8U);

    //将两个初始的mask复制到对应的位置，mask1和mask2的对应位置
    Mat tmp = mask1_(Rect(tl1.x - unionTl_.x, tl1.y - unionTl_.y, mask1.cols, mask1.rows));
    warpMask1.copyTo(tmp);

    tmp = mask2_(Rect(tl2.x - unionTl_.x, tl2.y - unionTl_.y, mask2.cols, mask2.rows));
    warpMask2.copyTo(tmp);

    // find both images contour masks
    //寻找对应的mask的边界
    contour1mask_ = Mat::zeros(unionSize_, CV_8U);
    contour2mask_ = Mat::zeros(unionSize_, CV_8U);

    for (int y = 0; y < unionSize_.height; ++y)
    {
        for (int x = 0; x < unionSize_.width; ++x)
        {
			if(mask2_(y, x)&&(index2!=2))//因为这个重叠区是给mask2的，所以如果index2在匹配树上低就赋值
			{
				if (mask1_(y, x) &&
					((x == 0 || !mask1_(y, x-1)) || (x == unionSize_.width-1 || !mask1_(y, x+1)) ||
					 (y == 0 || !mask1_(y-1, x)) || (y == unionSize_.height-1 || !mask1_(y+1, x))))
				{
					contour1mask_(y, x) = 255-index1;//warpMask1在warpMask2上的边界，但这个边界是要赋值给mask2的，所以减index1
				}
				else if(mask1_(y, x) )//赋值为125的像素在blend的时候是要被抹掉的，
					contour1mask_(y, x) = 125;//warpMask1在warpMask2上的重叠区域

			}
			if(mask1_(y, x)&&(index1!=2))
			{
				if (mask2_(y, x) &&
                ((x == 0 || !mask2_(y, x-1)) || (x == unionSize_.width-1 || !mask2_(y, x+1)) ||
                 (y == 0 || !mask2_(y-1, x)) || (y == unionSize_.height-1 || !mask2_(y+1, x))))
				{
					contour2mask_(y, x) = 255-index2;//warpMask2在warpMask1上的边界
				}
				else if(mask2_(y, x) )//
					contour2mask_(y, x) = 125;//warpMas2在warpMask1上的重叠区域
			}
            
        }
    }
	tmp = contour1mask_(Rect(tl2.x - unionTl_.x, tl2.y - unionTl_.y, mask2.cols, mask2.rows));
	//contour1mask_用2的矩形区去框，是为了找到这个1与2的重叠区相对于2的位置 
	for (int y = 0; y < tmp.rows; ++y){
			for (int x = 0; x < tmp.cols; ++x){				
				if(tmp.at<unsigned char>(y, x) > 0)
					mask2.at<unsigned char>(y, x) = tmp.at<unsigned char>(y, x);
			}
		}	//防止将mask2中已有的大于0的像素覆盖掉，只能一个一个赋值，不能直接用copy
	tmp = contour2mask_(Rect(tl1.x - unionTl_.x, tl1.y - unionTl_.y, mask1.cols, mask1.rows));
    for (int y = 0; y < tmp.rows; ++y){
			for (int x = 0; x < tmp.cols; ++x){				
				if(tmp.at<unsigned char>(y, x) > 0)
					mask1.at<unsigned char>(y, x) = tmp.at<unsigned char>(y, x);
			}
		}
}

