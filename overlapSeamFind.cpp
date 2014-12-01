#include "overlapSeamFind.h"


void  overlapSeamFinder::findSeam(vector<Mat> &warpImages, vector<Mat> &warpMask, vector<Mat> &seamMsk, vector<Point> &topleft)
{
	seamMsk.clear();
    for (unsigned int i=0;i<warpMask.size();i++){
		seamMsk.push_back(Mat::zeros(warpMask[i].size(),warpMask[i].type()));   //������Ҫ��ô���ƣ�����seamMask�ͺ�imgMask����һ���ڴ�����

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

    sort(pairs.begin(), pairs.end(), ImagePairLess(warpMasks, corners));   //����ÿ��֮������ĵ�ľ����������
    reverse(pairs.begin(), pairs.end());  //���������Ҳ���Ǿ����Զ����

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
        return; // there are no conflicts //�����ص�����û���ཻ

    unionTl_ = Point(std::min(tl1.x, tl2.x), std::min(tl1.y, tl2.y));

    unionBr_ = Point(std::max(tl1.x + warpMask1.cols, tl2.x + warpMask2.cols),
                     std::max(tl1.y + warpMask1.rows, tl2.y + warpMask2.rows));

    unionSize_ = Size(unionBr_.x - unionTl_.x, unionBr_.y - unionTl_.y);  //�����ཻ���������Ǻϲ�֮��Ĵ�С

    mask1_ = Mat::zeros(unionSize_, CV_8U);
    mask2_ = Mat::zeros(unionSize_, CV_8U);

    //��������ʼ��mask���Ƶ���Ӧ��λ�ã�mask1��mask2�Ķ�Ӧλ��
    Mat tmp = mask1_(Rect(tl1.x - unionTl_.x, tl1.y - unionTl_.y, mask1.cols, mask1.rows));
    warpMask1.copyTo(tmp);

    tmp = mask2_(Rect(tl2.x - unionTl_.x, tl2.y - unionTl_.y, mask2.cols, mask2.rows));
    warpMask2.copyTo(tmp);

    // find both images contour masks
    //Ѱ�Ҷ�Ӧ��mask�ı߽�
    contour1mask_ = Mat::zeros(unionSize_, CV_8U);
    contour2mask_ = Mat::zeros(unionSize_, CV_8U);

    for (int y = 0; y < unionSize_.height; ++y)
    {
        for (int x = 0; x < unionSize_.width; ++x)
        {
			if(mask2_(y, x)&&(index2!=2))//��Ϊ����ص����Ǹ�mask2�ģ��������index2��ƥ�����ϵ;͸�ֵ
			{
				if (mask1_(y, x) &&
					((x == 0 || !mask1_(y, x-1)) || (x == unionSize_.width-1 || !mask1_(y, x+1)) ||
					 (y == 0 || !mask1_(y-1, x)) || (y == unionSize_.height-1 || !mask1_(y+1, x))))
				{
					contour1mask_(y, x) = 255-index1;//warpMask1��warpMask2�ϵı߽磬������߽���Ҫ��ֵ��mask2�ģ����Լ�index1
				}
				else if(mask1_(y, x) )//��ֵΪ125��������blend��ʱ����Ҫ��Ĩ���ģ�
					contour1mask_(y, x) = 125;//warpMask1��warpMask2�ϵ��ص�����

			}
			if(mask1_(y, x)&&(index1!=2))
			{
				if (mask2_(y, x) &&
                ((x == 0 || !mask2_(y, x-1)) || (x == unionSize_.width-1 || !mask2_(y, x+1)) ||
                 (y == 0 || !mask2_(y-1, x)) || (y == unionSize_.height-1 || !mask2_(y+1, x))))
				{
					contour2mask_(y, x) = 255-index2;//warpMask2��warpMask1�ϵı߽�
				}
				else if(mask2_(y, x) )//
					contour2mask_(y, x) = 125;//warpMas2��warpMask1�ϵ��ص�����
			}
            
        }
    }
	tmp = contour1mask_(Rect(tl2.x - unionTl_.x, tl2.y - unionTl_.y, mask2.cols, mask2.rows));
	//contour1mask_��2�ľ�����ȥ����Ϊ���ҵ����1��2���ص��������2��λ�� 
	for (int y = 0; y < tmp.rows; ++y){
			for (int x = 0; x < tmp.cols; ++x){				
				if(tmp.at<unsigned char>(y, x) > 0)
					mask2.at<unsigned char>(y, x) = tmp.at<unsigned char>(y, x);
			}
		}	//��ֹ��mask2�����еĴ���0�����ظ��ǵ���ֻ��һ��һ����ֵ������ֱ����copy
	tmp = contour2mask_(Rect(tl1.x - unionTl_.x, tl1.y - unionTl_.y, mask1.cols, mask1.rows));
    for (int y = 0; y < tmp.rows; ++y){
			for (int x = 0; x < tmp.cols; ++x){				
				if(tmp.at<unsigned char>(y, x) > 0)
					mask1.at<unsigned char>(y, x) = tmp.at<unsigned char>(y, x);
			}
		}
}

