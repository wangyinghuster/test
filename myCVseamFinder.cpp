#include "myCVseamFinder.h"
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include <map>
using namespace std;
using namespace cv;

static inline float normL2(const Point3f& a)
{
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

static inline float normL2(const Point3f& a, const Point3f& b)
{
    return normL2(a - b);
}

static bool overlapRoi(Point tl1, Point tl2, Size sz1, Size sz2, Rect &roi)
{
    int x_tl = max(tl1.x, tl2.x);
    int y_tl = max(tl1.y, tl2.y);
    int x_br = min(tl1.x + sz1.width, tl2.x + sz2.width);
    int y_br = min(tl1.y + sz1.height, tl2.y + sz2.height);
    if (x_tl < x_br && y_tl < y_br)
    {
        roi = Rect(x_tl, y_tl, x_br - x_tl, y_br - y_tl);
        return true;
    }
    return false;
}

static inline int sqr(int x) { return x * x; }
static inline float sqr(float x) { return x * x; }
static inline double sqr(double x) { return x * x; }

myCVseamFinder::myCVseamFinder()
{
}

myCVdpSeamFinder::myCVdpSeamFinder(CostFunction costFunc) : costFunc_(costFunc) {}

void myCVdpSeamFinder::findSeam(vector<Mat> &warpImages, vector<Mat> &warpMask, vector<Mat> &seamMsk, vector<Point> &topleft)
{

    vector<Mat> imgF;
    imgF.resize(warpImages.size());
	seamMsk.clear();
    for (unsigned int i=0;i<warpMask.size();i++){
        warpImages[i].convertTo(imgF[i],CV_32F); //这里需要转换图像深度
        seamMsk.push_back(warpMask[i].clone());   //这里需要这么复制，否则seamMask就和imgMask共用一块内存区域。

    }

    find(imgF,topleft,seamMsk);

}

void myCVdpSeamFinder::find(const vector<Mat> &src, const vector<Point> &corners, vector<Mat> &masks)
{

    if (src.size() == 0)
        return;

    vector<pair<size_t, size_t> > pairs;

    for (size_t i = 0; i+1 < src.size(); ++i)
        for (size_t j = i+1; j < src.size(); ++j)
            pairs.push_back(make_pair(i, j));

    sort(pairs.begin(), pairs.end(), ImagePairLess(src, corners));   //按照每对之间的中心点的距离进行排序
    reverse(pairs.begin(), pairs.end());  //逆序过来，也就是距离从远到近

    for (size_t i = 0; i < pairs.size(); ++i)
    {
        size_t i0 = pairs[i].first, i1 = pairs[i].second;
        process(src[i0], src[i1], corners[i0], corners[i1], masks[i0], masks[i1]);
    }

}


void myCVdpSeamFinder::process(
        const Mat &image1, const Mat &image2, Point tl1, Point tl2,
        Mat &mask1, Mat &mask2)
{
    CV_Assert(image1.size() == mask1.size());
    CV_Assert(image2.size() == mask2.size());

    Point intersectTl(std::max(tl1.x, tl2.x), std::max(tl1.y, tl2.y));

    Point intersectBr(std::min(tl1.x + image1.cols, tl2.x + image2.cols),
                      std::min(tl1.y + image1.rows, tl2.y + image2.rows));

    if (intersectTl.x >= intersectBr.x || intersectTl.y >= intersectBr.y)
        return; // there are no conflicts //计算重叠区域，没有相交

    unionTl_ = Point(std::min(tl1.x, tl2.x), std::min(tl1.y, tl2.y));

    unionBr_ = Point(std::max(tl1.x + image1.cols, tl2.x + image2.cols),
                     std::max(tl1.y + image1.rows, tl2.y + image2.rows));

    unionSize_ = Size(unionBr_.x - unionTl_.x, unionBr_.y - unionTl_.y);  //若有相交，则计算的是合并之后的大小

    mask1_ = Mat::zeros(unionSize_, CV_8U);
    mask2_ = Mat::zeros(unionSize_, CV_8U);

    //将两个初始的mask复制到对应的位置，mask1和mask2的对应位置
    Mat tmp = mask1_(Rect(tl1.x - unionTl_.x, tl1.y - unionTl_.y, mask1.cols, mask1.rows));
    mask1.copyTo(tmp);

    tmp = mask2_(Rect(tl2.x - unionTl_.x, tl2.y - unionTl_.y, mask2.cols, mask2.rows));
    mask2.copyTo(tmp);

    // find both images contour masks
    //寻找对应的mask的边界
    contour1mask_ = Mat::zeros(unionSize_, CV_8U);
    contour2mask_ = Mat::zeros(unionSize_, CV_8U);

    for (int y = 0; y < unionSize_.height; ++y)
    {
        for (int x = 0; x < unionSize_.width; ++x)
        {
            if (mask1_(y, x) &&
                ((x == 0 || !mask1_(y, x-1)) || (x == unionSize_.width-1 || !mask1_(y, x+1)) ||
                 (y == 0 || !mask1_(y-1, x)) || (y == unionSize_.height-1 || !mask1_(y+1, x))))
            {
                contour1mask_(y, x) = 255;
            }

            if (mask2_(y, x) &&
                ((x == 0 || !mask2_(y, x-1)) || (x == unionSize_.width-1 || !mask2_(y, x+1)) ||
                 (y == 0 || !mask2_(y-1, x)) || (y == unionSize_.height-1 || !mask2_(y+1, x))))
            {
                contour2mask_(y, x) = 255;
            }
        }
    }

	//找出连通区域
    findComponents();

    findEdges();

    resolveConflicts(image1, image2, tl1, tl2, mask1, mask2);
}


void myCVdpSeamFinder::findComponents()
{
    // label all connected components and get information about them

    ncomps_ = 0;
    labels_.create(unionSize_);
    states_.clear();
    tls_.clear();
    brs_.clear();
    contours_.clear();

    //用mask来寻找相交的区域，并作标记，区分两者都有的，区分只有一边的，区分两个都没有的
	//等等，这里用的是y,x，转置了？不是，mat_类型重载的operater()是先rows和cols，所以访问x,y点是用，（y，x）
    for (int y = 0; y < unionSize_.height; ++y)
    {
        for (int x = 0; x < unionSize_.width; ++x)
        {
            if (mask1_(y, x) && mask2_(y, x))
                labels_(y, x) = numeric_limits<int>::max();
            else if (mask1_(y, x))
                labels_(y, x) = numeric_limits<int>::max()-1;
            else if (mask2_(y, x))
                labels_(y, x) = numeric_limits<int>::max()-2;
            else
                labels_(y, x) = 0;
        }
    }

   
    for (int y = 0; y < unionSize_.height; ++y)
    {
        for (int x = 0; x < unionSize_.width; ++x)
        {
			//将像素展开成一行，作状态标记?不是，原因见下
            if (labels_(y, x) >= numeric_limits<int>::max()-2)  //只有第一次进入连通域的某个像素的时候会执行的if区块
            {
                if (labels_(y, x) == numeric_limits<int>::max())
                    states_.push_back(INTERS);  //两者mask重叠的区域
                else if (labels_(y, x) == numeric_limits<int>::max()-1)
                    states_.push_back(FIRST);   //只有第一副图的区域
                else if (labels_(y, x) == numeric_limits<int>::max()-2)
                    states_.push_back(SECOND);  //只有第二副图的区域

                floodFill(labels_, Point(x, y), ++ncomps_);  //只有连通域才会被填充
                tls_.push_back(Point(x, y));
                brs_.push_back(Point(x+1, y+1));  //连通域大小的初始化
                contours_.push_back(vector<Point>());
            }

			//这里为什么要这么写？这两个if不是相同的意义么？
			//不是铁定是相等的么？
			//不是，可以这么理解，在连通域的第一个点会进入上一个if区块，但是经过floodFill“污染”之后，所有与这个点联通的点都变成了ncomps_的标记的值了。
			//所以此后在此连通域内的像素就会进入这个区块了
			//并且这个区块的颜色正好对应着tls_、brs_、contours_索引的值减一
            if (labels_(y, x))
            {
                int l = labels_(y, x);
                int ci = l-1;

				
                tls_[ci].x = std::min(tls_[ci].x, x);
                tls_[ci].y = std::min(tls_[ci].y, y);
                brs_[ci].x = std::max(brs_[ci].x, x+1);
                brs_[ci].y = std::max(brs_[ci].y, y+1);

                if ((x == 0 || labels_(y, x-1) != l) || (x == unionSize_.width-1 || labels_(y, x+1) != l) ||   //检测到四周某个点不是连通域了，所以这个点是属于边界的。
                    (y == 0 || labels_(y-1, x) != l) || (y == unionSize_.height-1 || labels_(y+1, x) != l))
                {
                    contours_[ci].push_back(Point(x, y));
                }
            }
        }
    }

	//所以这个函数以上的功能是在label中标记出联通区域的区块，并且区块的值
    //是对应以下vector的索引,全局坐标（原点为unionSize的topLeft）
	//tls_、brs_连通区域的左上右下定点。contours_连通区域的边界点的集合。states_连通区域的状态值
}


void myCVdpSeamFinder::findEdges()
{
    // find edges between components

    map<pair<int, int>, int> wedges; // weighted edges

    for (int ci = 0; ci < ncomps_-1; ++ci)   //对两两的联通区域配对
    {
        for (int cj = ci+1; cj < ncomps_; ++cj)
        {
            wedges[make_pair(ci, cj)] = 0;
            wedges[make_pair(cj, ci)] = 0;
        }
    }

    for (int ci = 0; ci < ncomps_; ++ci)
    {
        for (size_t i = 0; i < contours_[ci].size(); ++i)
        {
            int x = contours_[ci][i].x;
            int y = contours_[ci][i].y;
            int l = ci + 1;//标记值比索引值大一

            //遍历每个边界的点，查找相邻的连通域，只要相邻，那么wedges的值++
            //所以weidges的值和相邻的连通域的共同边界长度成正比
            if (x > 0 && labels_(y, x-1) && labels_(y, x-1) != l)
            {
                wedges[make_pair(ci, labels_(y, x-1)-1)]++;
                wedges[make_pair(labels_(y, x-1)-1, ci)]++;
            }

            if (y > 0 && labels_(y-1, x) && labels_(y-1, x) != l)
            {
                wedges[make_pair(ci, labels_(y-1, x)-1)]++;
                wedges[make_pair(labels_(y-1, x)-1, ci)]++;
            }

            if (x < unionSize_.width-1 && labels_(y, x+1) && labels_(y, x+1) != l)
            {
                wedges[make_pair(ci, labels_(y, x+1)-1)]++;
                wedges[make_pair(labels_(y, x+1)-1, ci)]++;
            }

            if (y < unionSize_.height-1 && labels_(y+1, x) && labels_(y+1, x) != l)
            {
                wedges[make_pair(ci, labels_(y+1, x)-1)]++;
                wedges[make_pair(labels_(y+1, x)-1, ci)]++;
            }
        }
    }

    edges_.clear();

    for (int ci = 0; ci < ncomps_-1; ++ci)
    {
        for (int cj = ci+1; cj < ncomps_; ++cj)
        {
            //map的iterator获取的是一个pair对象，所以返回的是pair<pair<int,int>,int>
            //所以itr->second是int，itr->first是pair<int,int>
            map<pair<int, int>, int>::iterator itr = wedges.find(make_pair(ci, cj));
            if (itr != wedges.end() && itr->second > 0)
                edges_.insert(itr->first);

            itr = wedges.find(make_pair(cj, ci));
            if (itr != wedges.end() && itr->second > 0)
                edges_.insert(itr->first);
        }
    }
    //所以以上代码会生成一个集合edges_，包含了所有有相邻的连通区域的边界
}


void myCVdpSeamFinder::resolveConflicts(
        const Mat &image1, const Mat &image2, Point tl1, Point tl2, Mat &mask1, Mat &mask2)
{
    if (costFunc_ == COLOR_GRAD)
        computeGradients(image1, image2);

    // resolve conflicts between components

    bool hasConflict = true;
    while (hasConflict)
    {
        int c1 = 0, c2 = 0;
        hasConflict = false;

        for (set<pair<int, int> >::iterator itr = edges_.begin(); itr != edges_.end(); ++itr)
        {
            c1 = itr->first;
            c2 = itr->second;

            //！=的优先级比&&要高，states需要inter，并且没有inter|second or first 的状态
            //与inter相邻的区域一定没有inter标记
            if ((states_[c1] & INTERS) && (states_[c1] & (~INTERS)) != states_[c2])
            {
                hasConflict = true;
                break;
            }
        }

        if (hasConflict)
        {
            int l1 = c1+1, l2 = c2+1;

            if (hasOnlyOneNeighbor(c1))
            {
                // if the first components has only one adjacent component

                for (int y = tls_[c1].y; y < brs_[c1].y; ++y)
                    for (int x = tls_[c1].x; x < brs_[c1].x; ++x)
                        if (labels_(y, x) == l1)
                            labels_(y, x) = l2;  //将这个联通区块的标志标为c2的标志，即合并到c2中

                states_[c1] = states_[c2] == FIRST ? SECOND : FIRST; //这里为什么要让c1和c2的不同？
            }
            else
            {
                // if the first component has more than one adjacent component

                Point p1, p2;
                //下面这个函数找出邻接的连通域的边界中相对距离最远的两个，p1,p2是边界上最靠近边界重心的点
                if (getSeamTips(c1, c2, p1, p2))
                {
                    vector<Point> seam;
                    bool isHorizontalSeam;

                    if (estimateSeam(image1, image2, tl1, tl2, c1, p1, p2, seam, isHorizontalSeam))
                        updateLabelsUsingSeam(c1, c2, seam, isHorizontalSeam);
                }

                states_[c1] = states_[c2] == FIRST ? INTERS_SECOND : INTERS_FIRST;
            }

            const int c[] = {c1, c2};
            const int l[] = {l1, l2};

            for (int i = 0; i < 2; ++i)
            {
                // update information about the (i+1)-th component

                int x0 = tls_[c[i]].x, x1 = brs_[c[i]].x;
                int y0 = tls_[c[i]].y, y1 = brs_[c[i]].y;

                tls_[c[i]] = Point(numeric_limits<int>::max(), numeric_limits<int>::max());
                brs_[c[i]] = Point(numeric_limits<int>::min(), numeric_limits<int>::min());
                contours_[c[i]].clear();

                for (int y = y0; y < y1; ++y)
                {
                    for (int x = x0; x < x1; ++x)
                    {
                        if (labels_(y, x) == l[i])
                        {
                            tls_[c[i]].x = std::min(tls_[c[i]].x, x);
                            tls_[c[i]].y = std::min(tls_[c[i]].y, y);
                            brs_[c[i]].x = std::max(brs_[c[i]].x, x+1);
                            brs_[c[i]].y = std::max(brs_[c[i]].y, y+1);

                            if ((x == 0 || labels_(y, x-1) != l[i]) || (x == unionSize_.width-1 || labels_(y, x+1) != l[i]) ||
                                (y == 0 || labels_(y-1, x) != l[i]) || (y == unionSize_.height-1 || labels_(y+1, x) != l[i]))
                            {
                                contours_[c[i]].push_back(Point(x, y));
                            }
                        }
                    }
                }
            }

            // remove edges

            edges_.erase(make_pair(c1, c2));
            edges_.erase(make_pair(c2, c1));
        }
    }

    // update masks

    int dx1 = unionTl_.x - tl1.x, dy1 = unionTl_.y - tl1.y;
    int dx2 = unionTl_.x - tl2.x, dy2 = unionTl_.y - tl2.y;

    for (int y = 0; y < mask2.rows; ++y)
    {
        for (int x = 0; x < mask2.cols; ++x)
        {
             int l = labels_(y - dy2, x - dx2);
             if (l > 0 && (states_[l-1] & FIRST) && mask1.at<uchar>(y - dy2 + dy1, x - dx2 + dx1))
                mask2.at<uchar>(y, x) = 0;
        }
    }

    for (int y = 0; y < mask1.rows; ++y)
    {
        for (int x = 0; x < mask1.cols; ++x)
        {
             int l = labels_(y - dy1, x - dx1);
             if (l > 0 && (states_[l-1] & SECOND) && mask2.at<uchar>(y - dy1 + dy2, x - dx1 + dx2))
                mask1.at<uchar>(y, x) = 0;
        }
    }
}


void myCVdpSeamFinder::computeGradients(const Mat &image1, const Mat &image2)
{
    CV_Assert(image1.channels() == 3 || image1.channels() == 4);
    CV_Assert(image2.channels() == 3 || image2.channels() == 4);
    CV_Assert(costFunction() == COLOR_GRAD);

    Mat gray;

    if (image1.channels() == 3)
        cvtColor(image1, gray, CV_BGR2GRAY);
    else if (image1.channels() == 4)
        cvtColor(image1, gray, CV_BGRA2GRAY);

    Sobel(gray, gradx1_, CV_32F, 1, 0);
    Sobel(gray, grady1_, CV_32F, 0, 1);

    if (image2.channels() == 3)
        cvtColor(image2, gray, CV_BGR2GRAY);
    else if (image2.channels() == 4)
        cvtColor(image2, gray, CV_BGRA2GRAY);

    Sobel(gray, gradx2_, CV_32F, 1, 0);
    Sobel(gray, grady2_, CV_32F, 0, 1);
}


bool myCVdpSeamFinder::hasOnlyOneNeighbor(int comp)
{
    //查找c1使得诸如 <c1，c2>这样的键值对，是不是在edges_中只有唯一一个c2与之对应
    //iterator lower_bound( const key_type &key ): 返回一个迭代器，指向键值>= key的第一个元素。
    //iterator upper_bound( const key_type &key ):返回一个迭代器，指向键值> key的第一个元素。

    set<pair<int, int> >::iterator begin, end;
    begin = lower_bound(edges_.begin(), edges_.end(), make_pair(comp, numeric_limits<int>::min()));
    end = upper_bound(edges_.begin(), edges_.end(), make_pair(comp, numeric_limits<int>::max()));
    return ++begin == end;
}

//这个函数判断所给的点是不是距离contourMask两个像素以内
bool myCVdpSeamFinder::closeToContour(int y, int x, const Mat_<uchar> &contourMask)
{
    const int rad = 2;

    for (int dy = -rad; dy <= rad; ++dy)
    {
        if (y + dy >= 0 && y + dy < unionSize_.height)
        {
            for (int dx = -rad; dx <= rad; ++dx)
            {
                if (x + dx >= 0 && x + dx < unionSize_.width &&
                    contourMask(y + dy, x + dx))
                {
                    return true;
                }
            }
        }
    }

    return false;
}


bool myCVdpSeamFinder::getSeamTips(int comp1, int comp2, Point &p1, Point &p2)
{
    CV_Assert(states_[comp1] & INTERS);

    // find special points

    vector<Point> specialPoints;
    int l2 = comp2+1;

    //这个也是全局的坐标
    for (size_t i = 0; i < contours_[comp1].size(); ++i)
    {
        int x = contours_[comp1][i].x;
        int y = contours_[comp1][i].y;

        if (closeToContour(y, x, contour1mask_) &&
            closeToContour(y, x, contour2mask_) &&
            ((x > 0 && labels_(y, x-1) == l2) ||
             (y > 0 && labels_(y-1, x) == l2) ||
             (x < unionSize_.width-1 && labels_(y, x+1) == l2) ||
             (y < unionSize_.height-1 && labels_(y+1, x) == l2)))
        {
            //在c1的边界中找出与c2共有的边界，且这个边界还应该在之前mask的边界附近
            specialPoints.push_back(Point(x, y));
        }
    }

    if (specialPoints.size() < 2)
        return false;

    // find clusters

    vector<int> labels;
    cv::partition(specialPoints, labels, ClosePoints(10));  //分类函数，将specialPoints分成几类，类是由labels标记的。
                                                            //不过意义在哪里？找出被分割的边界？比如十字交叉类型的？

    int nlabels = *max_element(labels.begin(), labels.end()) + 1;
    if (nlabels < 2)
        return false;

    vector<Point> sum(nlabels);
    vector<vector<Point> > points(nlabels);

    //归类，并且统计点的向量和，这样就可以求取点的重心
    for (size_t i = 0; i < specialPoints.size(); ++i)
    {
        sum[labels[i]] += specialPoints[i];
        points[labels[i]].push_back(specialPoints[i]);
    }

    // select two most distant clusters

    int idx[2] = {-1,-1};
    double maxDist = -numeric_limits<double>::max();

    //计算类的类心距离，找出距离最远的两个类
    for (int i = 0; i < nlabels-1; ++i)
    {
        for (int j = i+1; j < nlabels; ++j)
        {
            double size1 = static_cast<double>(points[i].size()), size2 = static_cast<double>(points[j].size());
            double cx1 = cvRound(sum[i].x / size1), cy1 = cvRound(sum[i].y / size1);
            double cx2 = cvRound(sum[j].x / size2), cy2 = cvRound(sum[j].y / size1);

            double dist = (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2);
            if (dist > maxDist)
            {
                maxDist = dist;
                idx[0] = i;
                idx[1] = j;
            }
        }
    }

    // select two points closest to the clusters' centers

    Point p[2];
    //分别找出距离那两个类的类心最近的两个点
    for (int i = 0; i < 2; ++i)
    {
        double size = static_cast<double>(points[idx[i]].size());
        double cx = cvRound(sum[idx[i]].x / size);
        double cy = cvRound(sum[idx[i]].y / size);

        size_t closest = points[idx[i]].size();
        double minDist = numeric_limits<double>::max();

        for (size_t j = 0; j < points[idx[i]].size(); ++j)
        {
            double dist = (points[idx[i]][j].x - cx) * (points[idx[i]][j].x - cx) +
                          (points[idx[i]][j].y - cy) * (points[idx[i]][j].y - cy);
            if (dist < minDist)
            {
                minDist = dist;
                closest = j;
            }
        }

        p[i] = points[idx[i]][closest];
    }

    p1 = p[0];
    p2 = p[1];
    return true;
}


namespace
{

template <typename T>
float diffL2Square3(const Mat &image1, int y1, int x1, const Mat &image2, int y2, int x2)
{
    const T *r1 = image1.ptr<T>(y1);
    const T *r2 = image2.ptr<T>(y2);
    return static_cast<float>(sqr(r1[3*x1] - r2[3*x2]) + sqr(r1[3*x1+1] - r2[3*x2+1]) +
                              sqr(r1[3*x1+2] - r2[3*x2+2]));
}


template <typename T>
float diffL2Square4(const Mat &image1, int y1, int x1, const Mat &image2, int y2, int x2)
{
    const T *r1 = image1.ptr<T>(y1);
    const T *r2 = image2.ptr<T>(y2);
    return static_cast<float>(sqr(r1[4*x1] - r2[4*x2]) + sqr(r1[4*x1+1] - r2[4*x2+1]) +
                              sqr(r1[4*x1+2] - r2[4*x2+2]));
}

} // namespace

void myCVdpSeamFinder::computeCosts(
        const Mat &image1, const Mat &image2, Point tl1, Point tl2,
        int comp, Mat_<float> &costV, Mat_<float> &costH)
{
    CV_Assert(states_[comp] & INTERS);

    // compute costs

    float (*diff)(const Mat&, int, int, const Mat&, int, int) = 0;   //函数模板的指针，方便对不同的类型作用
    if (image1.type() == CV_32FC3 && image2.type() == CV_32FC3)
        diff = diffL2Square3<float>;
    else if (image1.type() == CV_8UC3 && image2.type() == CV_8UC3)
        diff = diffL2Square3<uchar>;
    else if (image1.type() == CV_32FC4 && image2.type() == CV_32FC4)
        diff = diffL2Square4<float>;
    else if (image1.type() == CV_8UC4 && image2.type() == CV_8UC4)
        diff = diffL2Square4<uchar>;
    else
        CV_Error(CV_StsBadArg, "both images must have CV_32FC3(4) or CV_8UC3(4) type");

    int l = comp+1;
    Rect roi(tls_[comp], brs_[comp]);

    //这里，tl之类的如果为负值会不会有问题？不会有问题，因为unionTl也有可能是负值
    int dx1 = unionTl_.x - tl1.x, dy1 = unionTl_.y - tl1.y;
    int dx2 = unionTl_.x - tl2.x, dy2 = unionTl_.y - tl2.y;

    const float badRegionCost = normL2(Point3f(255.f, 255.f, 255.f),
                                       Point3f(0.f, 0.f, 0.f));

    costV.create(roi.height, roi.width+1);

    for (int y = roi.y; y < roi.br().y; ++y)
    {
        for (int x = roi.x; x < roi.br().x+1; ++x)
        {
            if (labels_(y, x) == l && x > 0 && labels_(y, x-1) == l)  //向左的梯度？V
            {
                float costColor = (diff(image1, y + dy1, x + dx1 - 1, image2, y + dy2, x + dx2) +
                                   diff(image1, y + dy1, x + dx1, image2, y + dy2, x + dx2 - 1)) / 2;
                if (costFunc_ == COLOR)
                    costV(y - roi.y, x - roi.x) = costColor;
                else if (costFunc_ == COLOR_GRAD)
                {
                    float costGrad = std::abs(gradx1_(y + dy1, x + dx1)) + std::abs(gradx1_(y + dy1, x + dx1 - 1)) +
                                     std::abs(gradx2_(y + dy2, x + dx2)) + std::abs(gradx2_(y + dy2, x + dx2 - 1)) + 1.f;
                    costV(y - roi.y, x - roi.x) = costColor / costGrad;
                }
            }
            else
                costV(y - roi.y, x - roi.x) = badRegionCost;
        }
    }

    costH.create(roi.height+1, roi.width);

    for (int y = roi.y; y < roi.br().y+1; ++y)
    {
        for (int x = roi.x; x < roi.br().x; ++x)
        {
            if (labels_(y, x) == l && y > 0 && labels_(y-1, x) == l)   //向上的梯度？H
            {
                float costColor = (diff(image1, y + dy1 - 1, x + dx1, image2, y + dy2, x + dx2) +
                                   diff(image1, y + dy1, x + dx1, image2, y + dy2 - 1, x + dx2)) / 2;
                if (costFunc_ == COLOR)
                    costH(y - roi.y, x - roi.x) = costColor;
                else if (costFunc_ == COLOR_GRAD)
                {
                    float costGrad = std::abs(grady1_(y + dy1, x + dx1)) + std::abs(grady1_(y + dy1 - 1, x + dx1)) +
                                     std::abs(grady2_(y + dy2, x + dx2)) + std::abs(grady2_(y + dy2 - 1, x + dx2)) + 1.f;
                    costH(y - roi.y, x - roi.x) = costColor / costGrad;
                }
            }
            else
                costH(y - roi.y, x - roi.x) = badRegionCost;
        }
    }
}


bool myCVdpSeamFinder::estimateSeam(
        const Mat &image1, const Mat &image2, Point tl1, Point tl2, int comp,
        Point p1, Point p2, vector<Point> &seam, bool &isHorizontal)
{
    CV_Assert(states_[comp] & INTERS);

    Mat_<float> costV, costH;
    computeCosts(image1, image2, tl1, tl2, comp, costV, costH);

    Rect roi(tls_[comp], brs_[comp]);
    Point src = p1 - roi.tl();  //计算起点和终点，不过这个到底是怎么分布的呢？
    Point dst = p2 - roi.tl();
    int l = comp+1;

    // estimate seam direction

    bool swapped = false;
    isHorizontal = std::abs(dst.x - src.x) > std::abs(dst.y - src.y);

    if (isHorizontal)
    {
        if (src.x > dst.x)
        {
            std::swap(src, dst);
            swapped = true;
        }
    }
    else if (src.y > dst.y)
    {
        swapped = true;
        std::swap(src, dst);
    }

    // find optimal control

    Mat_<uchar> control = Mat::zeros(roi.size(), CV_8U);
    Mat_<uchar> reachable = Mat::zeros(roi.size(), CV_8U);
    Mat_<float> cost = Mat::zeros(roi.size(), CV_32F);

    reachable(src) = 1;
    cost(src) = 0.f;

    int nsteps;
    pair<float, int> steps[3];

    if (isHorizontal)
    {
        for (int x = src.x+1; x <= dst.x; ++x)
        {
            for (int y = 0; y < roi.height; ++y)
            {
                // seam follows along upper side of pixels

                nsteps = 0;

                if (labels_(y + roi.y, x + roi.x) == l)
                {
                    if (reachable(y, x-1))
                        steps[nsteps++] = make_pair(cost(y, x-1) + costH(y, x-1), 1);
                    if (y > 0 && reachable(y-1, x-1))
                        steps[nsteps++] = make_pair(cost(y-1, x-1) + costH(y-1, x-1) + costV(y-1, x), 2);
                    if (y < roi.height-1 && reachable(y+1, x-1))
                        steps[nsteps++] = make_pair(cost(y+1, x-1) + costH(y+1, x-1) + costV(y, x), 3);
                }

                if (nsteps)
                {
                    pair<float, int> opt = *min_element(steps, steps + nsteps);
                    cost(y, x) = opt.first;
                    control(y, x) = (uchar)opt.second;
                    reachable(y, x) = 255;
                }
            }
        }
    }
    else
    {
        for (int y = src.y+1; y <= dst.y; ++y)
        {
            for (int x = 0; x < roi.width; ++x)
            {
                // seam follows along left side of pixels

                nsteps = 0;

                if (labels_(y + roi.y, x + roi.x) == l)
                {
                    if (reachable(y-1, x))
                        steps[nsteps++] = make_pair(cost(y-1, x) + costV(y-1, x), 1);
                    if (x > 0 && reachable(y-1, x-1))
                        steps[nsteps++] = make_pair(cost(y-1, x-1) + costV(y-1, x-1) + costH(y, x-1), 2);
                    if (x < roi.width-1 && reachable(y-1, x+1))
                        steps[nsteps++] = make_pair(cost(y-1, x+1) + costV(y-1, x+1) + costH(y, x), 3);
                }

                if (nsteps)
                {
                    pair<float, int> opt = *min_element(steps, steps + nsteps);
                    cost(y, x) = opt.first;
                    control(y, x) = (uchar)opt.second;
                    reachable(y, x) = 255;
                }
            }
        }
    }

    if (!reachable(dst))
        return false;

    // restore seam

    Point p = dst;
    seam.clear();
    seam.push_back(p + roi.tl());

    if (isHorizontal)
    {
        for (; p.x != src.x; seam.push_back(p + roi.tl()))
        {
            if (control(p) == 2) p.y--;
            else if (control(p) == 3) p.y++;
            p.x--;
        }
    }
    else
    {
        for (; p.y != src.y; seam.push_back(p + roi.tl()))
        {
            if (control(p) == 2) p.x--;
            else if (control(p) == 3) p.x++;
            p.y--;
        }
    }

    if (!swapped)
        reverse(seam.begin(), seam.end());

    CV_Assert(seam.front() == p1);
    CV_Assert(seam.back() == p2);
    return true;
}


void myCVdpSeamFinder::updateLabelsUsingSeam(
        int comp1, int comp2, const vector<Point> &seam, bool isHorizontalSeam)
{
    Mat_<int> mask = Mat::zeros(brs_[comp1].y - tls_[comp1].y,
                                brs_[comp1].x - tls_[comp1].x, CV_32S);

    for (size_t i = 0; i < contours_[comp1].size(); ++i)
        mask(contours_[comp1][i] - tls_[comp1]) = 255;

    for (size_t i = 0; i < seam.size(); ++i)
        mask(seam[i] - tls_[comp1]) = 255;

    // find connected components after seam carving
    //cout <<"new:"<<endl;
    int l1 = comp1+1, l2 = comp2+1;

    int ncomps = 0;

    for (int y = 0; y < mask.rows; ++y)
        for (int x = 0; x < mask.cols; ++x)
            if (!mask(y, x) && labels_(y + tls_[comp1].y, x + tls_[comp1].x) == l1)
                floodFill(mask, Point(x, y), ++ncomps);

    for (size_t i = 0; i < contours_[comp1].size(); ++i)
    {
        int x = contours_[comp1][i].x - tls_[comp1].x;
        int y = contours_[comp1][i].y - tls_[comp1].y;

        bool ok = false;
        static const int dx[] = {-1, +1, 0, 0, -1, +1, -1, +1};
        static const int dy[] = {0, 0, -1, +1, -1, -1, +1, +1};

        for (int j = 0; j < 8; ++j)
        {
            int c = x + dx[j];
            int r = y + dy[j];

            if (c >= 0 && c < mask.cols && r >= 0 && r < mask.rows &&
                mask(r, c) && mask(r, c) != 255)
            {
                ok = true;
                mask(y, x) = mask(r, c);
            }
        }

        if (!ok)
            mask(y, x) = 0;
    }

    if (isHorizontalSeam)
    {
        for (size_t i = 0; i < seam.size(); ++i)
        {
            int x = seam[i].x - tls_[comp1].x;
            int y = seam[i].y - tls_[comp1].y;

            if (y < mask.rows-1 && mask(y+1, x) && mask(y+1, x) != 255)
                mask(y, x) = mask(y+1, x);
            else
                mask(y, x) = 0;
        }
    }
    else
    {
        for (size_t i = 0; i < seam.size(); ++i)
        {
            int x = seam[i].x - tls_[comp1].x;
            int y = seam[i].y - tls_[comp1].y;

            if (x < mask.cols-1 && mask(y, x+1) && mask(y, x+1) != 255)
                mask(y, x) = mask(y, x+1);
            else
                mask(y, x) = 0;
        }
    }

    // find new components connected with the second component and
    // with other components except the ones we are working with


    map<int, int> connect2;
    map<int, int> connectOther;

    for (int i = 0; i <= ncomps; ++i) //把1改成了0
    {
        connect2.insert(make_pair(i, 0));
        connectOther.insert(make_pair(i, 0));
    }

    for (size_t i = 0; i < contours_[comp1].size(); ++i)
    {
        int x = contours_[comp1][i].x;
        int y = contours_[comp1][i].y;

        if ((x > 0 && labels_(y, x-1) == l2) ||
            (y > 0 && labels_(y-1, x) == l2) ||
            (x < unionSize_.width-1 && labels_(y, x+1) == l2) ||
            (y < unionSize_.height-1 && labels_(y+1, x) == l2))
        {
            connect2[mask(y - tls_[comp1].y, x - tls_[comp1].x)]++;
        }

        if ((x > 0 && labels_(y, x-1) != l1 && labels_(y, x-1) != l2) ||
            (y > 0 && labels_(y-1, x) != l1 && labels_(y-1, x) != l2) ||
            (x < unionSize_.width-1 && labels_(y, x+1) != l1 && labels_(y, x+1) != l2) ||
            (y < unionSize_.height-1 && labels_(y+1, x) != l1 && labels_(y+1, x) != l2))
        {
            connectOther[mask(y - tls_[comp1].y, x - tls_[comp1].x)]++;
        }
    }

    vector<int> isAdjComp(ncomps + 1, 0);

    //cout << "here7" <<endl;
    for (map<int, int>::iterator itr = connect2.begin(); itr != connect2.end(); ++itr)
    {
        //cout << "here7.125" <<endl;
        double len = static_cast<double>(contours_[comp1].size());
        //cout << "here7.25" <<endl;
        //cout << len<<","<<itr->first <<","<< itr->second <<","<<(ncomps+1)<<endl;
        //cout << connectOther.find(itr->first)->second<<","<<endl;  //就是这个地方会出错！！所以我在之前的for循环哪里把1改成了0
        isAdjComp[itr->first] = itr->second / len > 0.05 && connectOther.find(itr->first)->second / len < 0.1;  //这句话会莫名其妙的报invalid parameter的错误
        //cout << "here7.49"<<endl;
    }

    //cout << "here7.5" <<endl;
    // update labels

    for (int y = 0; y < mask.rows; ++y)
        for (int x = 0; x < mask.cols; ++x)
            if (mask(y, x) && isAdjComp[mask(y, x)])
                labels_(y + tls_[comp1].y, x + tls_[comp1].x) = l2;

}

/*
void myCVpairwiseSeamFinder::find(const vector<Mat> &src, const vector<Point> &corners,
                              vector<Mat> &masks)
{
    if (src.size() == 0)
        return;

    images_ = src;
    sizes_.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        sizes_[i] = src[i].size();
    corners_ = corners;
    masks_ = masks;
    run();

}


void myCVpairwiseSeamFinder::run()
{
    for (size_t i = 0; i < sizes_.size() - 1; ++i)
    {
        for (size_t j = i + 1; j < sizes_.size(); ++j)
        {
            Rect roi;
            if (overlapRoi(corners_[i], corners_[j], sizes_[i], sizes_[j], roi))
                findInPair(i, j, roi);
        }
    }
}


void myCVvoronoiSeamFinder::find(const vector<Size> &sizes, const vector<Point> &corners,
                             vector<Mat> &masks)
{

    if (sizes.size() == 0)
        return;

    sizes_ = sizes;
    corners_ = corners;
    masks_ = masks;
    run();
}


void myCVvoronoiSeamFinder::findInPair(size_t first, size_t second, Rect roi)
{
    const int gap = 10;
    Mat submask1(roi.height + 2 * gap, roi.width + 2 * gap, CV_8U);
    Mat submask2(roi.height + 2 * gap, roi.width + 2 * gap, CV_8U);

    Size img1 = sizes_[first], img2 = sizes_[second];
    Mat mask1 = masks_[first], mask2 = masks_[second];
    Point tl1 = corners_[first], tl2 = corners_[second];

    // Cut submasks with some gap
    for (int y = -gap; y < roi.height + gap; ++y)
    {
        for (int x = -gap; x < roi.width + gap; ++x)
        {
            int y1 = roi.y - tl1.y + y;
            int x1 = roi.x - tl1.x + x;
            if (y1 >= 0 && x1 >= 0 && y1 < img1.height && x1 < img1.width)
                submask1.at<uchar>(y + gap, x + gap) = mask1.at<uchar>(y1, x1);
            else
                submask1.at<uchar>(y + gap, x + gap) = 0;

            int y2 = roi.y - tl2.y + y;
            int x2 = roi.x - tl2.x + x;
            if (y2 >= 0 && x2 >= 0 && y2 < img2.height && x2 < img2.width)
                submask2.at<uchar>(y + gap, x + gap) = mask2.at<uchar>(y2, x2);
            else
                submask2.at<uchar>(y + gap, x + gap) = 0;
        }
    }

    Mat collision = (submask1 != 0) & (submask2 != 0);
    Mat unique1 = submask1.clone(); unique1.setTo(0, collision);
    Mat unique2 = submask2.clone(); unique2.setTo(0, collision);

    Mat dist1, dist2;
    distanceTransform(unique1 == 0, dist1, CV_DIST_L1, 3);
    distanceTransform(unique2 == 0, dist2, CV_DIST_L1, 3);

    Mat seam = dist1 < dist2;

    for (int y = 0; y < roi.height; ++y)
    {
        for (int x = 0; x < roi.width; ++x)
        {
            if (seam.at<uchar>(y + gap, x + gap))
                mask2.at<uchar>(roi.y - tl2.y + y, roi.x - tl2.x + x) = 0;
            else
                mask1.at<uchar>(roi.y - tl1.y + y, roi.x - tl1.x + x) = 0;
        }
    }
}

class myCVgraphCutSeamFinder::Impl : public myCVpairwiseSeamFinder
{
public:
    Impl(int cost_type, float terminal_cost, float bad_region_penalty)
        : cost_type_(cost_type), terminal_cost_(terminal_cost), bad_region_penalty_(bad_region_penalty) {}

    ~Impl() {}

    void find(const vector<Mat> &src, const vector<Point> &corners, vector<Mat> &masks);
    void findInPair(size_t first, size_t second, Rect roi);

private:
    void setGraphWeightsColor(const Mat &img1, const Mat &img2,
                              const Mat &mask1, const Mat &mask2, GCGraph<float> &graph);
    void setGraphWeightsColorGrad(const Mat &img1, const Mat &img2, const Mat &dx1, const Mat &dx2,
                                  const Mat &dy1, const Mat &dy2, const Mat &mask1, const Mat &mask2,
                                  GCGraph<float> &graph);

    vector<Mat> dx_, dy_;
    int cost_type_;
    float terminal_cost_;
    float bad_region_penalty_;
};


void myCVgraphCutSeamFinder::Impl::find(const vector<Mat> &src, const vector<Point> &corners,
                                    vector<Mat> &masks)
{
    // Compute gradients
    dx_.resize(src.size());
    dy_.resize(src.size());
    Mat dx, dy;
    for (size_t i = 0; i < src.size(); ++i)
    {
        CV_Assert(src[i].channels() == 3);
        Sobel(src[i], dx, CV_32F, 1, 0);
        Sobel(src[i], dy, CV_32F, 0, 1);
        dx_[i].create(src[i].size(), CV_32F);
        dy_[i].create(src[i].size(), CV_32F);
        for (int y = 0; y < src[i].rows; ++y)
        {
            const Point3f* dx_row = dx.ptr<Point3f>(y);
            const Point3f* dy_row = dy.ptr<Point3f>(y);
            float* dx_row_ = dx_[i].ptr<float>(y);
            float* dy_row_ = dy_[i].ptr<float>(y);
            for (int x = 0; x < src[i].cols; ++x)
            {
                dx_row_[x] = normL2(dx_row[x]);
                dy_row_[x] = normL2(dy_row[x]);
            }
        }
    }
    myCVpairwiseSeamFinder::find(src, corners, masks);
}


void myCVgraphCutSeamFinder::Impl::setGraphWeightsColor(const Mat &img1, const Mat &img2,
                                                    const Mat &mask1, const Mat &mask2, GCGraph<float> &graph)
{
    const Size img_size = img1.size();

    // Set terminal weights
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            int v = graph.addVtx();
            graph.addTermWeights(v, mask1.at<uchar>(y, x) ? terminal_cost_ : 0.f,
                                    mask2.at<uchar>(y, x) ? terminal_cost_ : 0.f);
        }
    }

    // Set regular edge weights
    const float weight_eps = 1.f;
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            int v = y * img_size.width + x;
            if (x < img_size.width - 1)
            {
                float weight = normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x)) +
                               normL2(img1.at<Point3f>(y, x + 1), img2.at<Point3f>(y, x + 1)) +
                               weight_eps;
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y, x + 1) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y, x + 1))
                    weight += bad_region_penalty_;
                graph.addEdges(v, v + 1, weight, weight);
            }
            if (y < img_size.height - 1)
            {
                float weight = normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x)) +
                               normL2(img1.at<Point3f>(y + 1, x), img2.at<Point3f>(y + 1, x)) +
                               weight_eps;
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y + 1, x) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y + 1, x))
                    weight += bad_region_penalty_;
                graph.addEdges(v, v + img_size.width, weight, weight);
            }
        }
    }
}


void myCVgraphCutSeamFinder::Impl::setGraphWeightsColorGrad(
        const Mat &img1, const Mat &img2, const Mat &dx1, const Mat &dx2,
        const Mat &dy1, const Mat &dy2, const Mat &mask1, const Mat &mask2,
        GCGraph<float> &graph)
{
    const Size img_size = img1.size();

    // Set terminal weights
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            int v = graph.addVtx();
            graph.addTermWeights(v, mask1.at<uchar>(y, x) ? terminal_cost_ : 0.f,
                                    mask2.at<uchar>(y, x) ? terminal_cost_ : 0.f);
        }
    }

    // Set regular edge weights
    const float weight_eps = 1.f;
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            int v = y * img_size.width + x;
            if (x < img_size.width - 1)
            {
                float grad = dx1.at<float>(y, x) + dx1.at<float>(y, x + 1) +
                             dx2.at<float>(y, x) + dx2.at<float>(y, x + 1) + weight_eps;
                float weight = (normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x)) +
                                normL2(img1.at<Point3f>(y, x + 1), img2.at<Point3f>(y, x + 1))) / grad +
                               weight_eps;
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y, x + 1) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y, x + 1))
                    weight += bad_region_penalty_;
                graph.addEdges(v, v + 1, weight, weight);
            }
            if (y < img_size.height - 1)
            {
                float grad = dy1.at<float>(y, x) + dy1.at<float>(y + 1, x) +
                             dy2.at<float>(y, x) + dy2.at<float>(y + 1, x) + weight_eps;
                float weight = (normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x)) +
                                normL2(img1.at<Point3f>(y + 1, x), img2.at<Point3f>(y + 1, x))) / grad +
                               weight_eps;
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y + 1, x) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y + 1, x))
                    weight += bad_region_penalty_;
                graph.addEdges(v, v + img_size.width, weight, weight);
            }
        }
    }
}


void myCVgraphCutSeamFinder::Impl::findInPair(size_t first, size_t second, Rect roi)
{
    Mat img1 = images_[first], img2 = images_[second];
    Mat dx1 = dx_[first], dx2 = dx_[second];
    Mat dy1 = dy_[first], dy2 = dy_[second];
    Mat mask1 = masks_[first], mask2 = masks_[second];
    Point tl1 = corners_[first], tl2 = corners_[second];

    const int gap = 10;
    Mat subimg1(roi.height + 2 * gap, roi.width + 2 * gap, CV_32FC3);
    Mat subimg2(roi.height + 2 * gap, roi.width + 2 * gap, CV_32FC3);
    Mat submask1(roi.height + 2 * gap, roi.width + 2 * gap, CV_8U);
    Mat submask2(roi.height + 2 * gap, roi.width + 2 * gap, CV_8U);
    Mat subdx1(roi.height + 2 * gap, roi.width + 2 * gap, CV_32F);
    Mat subdy1(roi.height + 2 * gap, roi.width + 2 * gap, CV_32F);
    Mat subdx2(roi.height + 2 * gap, roi.width + 2 * gap, CV_32F);
    Mat subdy2(roi.height + 2 * gap, roi.width + 2 * gap, CV_32F);

    // Cut subimages and submasks with some gap
    for (int y = -gap; y < roi.height + gap; ++y)
    {
        for (int x = -gap; x < roi.width + gap; ++x)
        {
            int y1 = roi.y - tl1.y + y;
            int x1 = roi.x - tl1.x + x;
            if (y1 >= 0 && x1 >= 0 && y1 < img1.rows && x1 < img1.cols)
            {
                subimg1.at<Point3f>(y + gap, x + gap) = img1.at<Point3f>(y1, x1);
                submask1.at<uchar>(y + gap, x + gap) = mask1.at<uchar>(y1, x1);
                subdx1.at<float>(y + gap, x + gap) = dx1.at<float>(y1, x1);
                subdy1.at<float>(y + gap, x + gap) = dy1.at<float>(y1, x1);
            }
            else
            {
                subimg1.at<Point3f>(y + gap, x + gap) = Point3f(0, 0, 0);
                submask1.at<uchar>(y + gap, x + gap) = 0;
                subdx1.at<float>(y + gap, x + gap) = 0.f;
                subdy1.at<float>(y + gap, x + gap) = 0.f;
            }

            int y2 = roi.y - tl2.y + y;
            int x2 = roi.x - tl2.x + x;
            if (y2 >= 0 && x2 >= 0 && y2 < img2.rows && x2 < img2.cols)
            {
                subimg2.at<Point3f>(y + gap, x + gap) = img2.at<Point3f>(y2, x2);
                submask2.at<uchar>(y + gap, x + gap) = mask2.at<uchar>(y2, x2);
                subdx2.at<float>(y + gap, x + gap) = dx2.at<float>(y2, x2);
                subdy2.at<float>(y + gap, x + gap) = dy2.at<float>(y2, x2);
            }
            else
            {
                subimg2.at<Point3f>(y + gap, x + gap) = Point3f(0, 0, 0);
                submask2.at<uchar>(y + gap, x + gap) = 0;
                subdx2.at<float>(y + gap, x + gap) = 0.f;
                subdy2.at<float>(y + gap, x + gap) = 0.f;
            }
        }
    }

    const int vertex_count = (roi.height + 2 * gap) * (roi.width + 2 * gap);
    const int edge_count = (roi.height - 1 + 2 * gap) * (roi.width + 2 * gap) +
                           (roi.width - 1 + 2 * gap) * (roi.height + 2 * gap);
    GCGraph<float> graph(vertex_count, edge_count);

    switch (cost_type_)
    {
    case myCVgraphCutSeamFinder::COST_COLOR:
        setGraphWeightsColor(subimg1, subimg2, submask1, submask2, graph);
        break;
    case myCVgraphCutSeamFinder::COST_COLOR_GRAD:
        setGraphWeightsColorGrad(subimg1, subimg2, subdx1, subdx2, subdy1, subdy2,
                                 submask1, submask2, graph);
        break;
    default:
        CV_Error(CV_StsBadArg, "unsupported pixel similarity measure");
    }

    graph.maxFlow();

    for (int y = 0; y < roi.height; ++y)
    {
        for (int x = 0; x < roi.width; ++x)
        {
            if (graph.inSourceSegment((y + gap) * (roi.width + 2 * gap) + x + gap))
            {
                if (mask1.at<uchar>(roi.y - tl1.y + y, roi.x - tl1.x + x))
                    mask2.at<uchar>(roi.y - tl2.y + y, roi.x - tl2.x + x) = 0;
            }
            else
            {
                if (mask2.at<uchar>(roi.y - tl2.y + y, roi.x - tl2.x + x))
                    mask1.at<uchar>(roi.y - tl1.y + y, roi.x - tl1.x + x) = 0;
            }
        }
    }
}


myCVgraphCutSeamFinder::myCVgraphCutSeamFinder(int cost_type, float terminal_cost, float bad_region_penalty)
    : impl_(new Impl(cost_type, terminal_cost, bad_region_penalty)) {}

myCVgraphCutSeamFinder::~myCVgraphCutSeamFinder() {}

void myCVgraphCutSeamFinder::findSeam(vector<Mat> &warpImages, vector<Mat> &warpMask, vector<Mat> &seamMsk, vector<Point> &topleft)
{

    int xmax=0,ymax=0;
    for (unsigned int i=0;i<warpMask.size();i++)
    {
        Point &p=topleft[i];
        int xT=p.x+warpMask[i].cols;
        int yT=p.y+warpMask[i].rows;
        xmax=xmax>xT?xmax:xT;
        ymax=ymax>yT?ymax:yT;
    }

    //计算resize参数的大小。这里需要计算一下resize，不然的话实在是太慢了。
    double seamMegapix=0.1; //这个参数，自己看看能不能改
    double seamScale;
    seamScale=sqrt(seamMegapix * 1e6 / (ymax*xmax));
    seamScale=1.0<seamScale?1.0:seamScale;  //计算resize参数，不让放大，只让缩小

    vector<Mat> imgF(warpMask.size());
    for (unsigned int i=0;i<warpMask.size();i++){
        warpImages[i].convertTo(imgF[i],CV_32F); //这里需要转换图像深度
        seamMsk.push_back(warpMask[i].clone());   //这里需要这么复制，否则seamMask就和imgMask共用一块内存区域。

        resize(imgF[i],imgF[i],Size(),seamScale,seamScale); //变换大小
        resize(seamMsk[i],seamMsk[i],Size(),seamScale,seamScale); //变换大小

    }

    find(imgF,topleft,seamMsk);   //这个会超出内存大小，所以还是得来看看

    for (unsigned int i=0;i<warpMask.size();i++){
        resize(seamMsk[i],seamMsk[i],warpMask[i].size()); //大小换回来

    }
}

void myCVgraphCutSeamFinder::find(const vector<Mat> &src, const vector<Point> &corners,
                              vector<Mat> &masks)
{
    impl_->find(src, corners, masks);
}


#ifdef NO_HAVE_OPENCV_GPU
void GraphCutSeamFinderGpu::find(const vector<Mat> &src, const vector<Point> &corners,
                                 vector<Mat> &masks)
{
    // Compute gradients
    dx_.resize(src.size());
    dy_.resize(src.size());
    Mat dx, dy;
    for (size_t i = 0; i < src.size(); ++i)
    {
        CV_Assert(src[i].channels() == 3);
        Sobel(src[i], dx, CV_32F, 1, 0);
        Sobel(src[i], dy, CV_32F, 0, 1);
        dx_[i].create(src[i].size(), CV_32F);
        dy_[i].create(src[i].size(), CV_32F);
        for (int y = 0; y < src[i].rows; ++y)
        {
            const Point3f* dx_row = dx.ptr<Point3f>(y);
            const Point3f* dy_row = dy.ptr<Point3f>(y);
            float* dx_row_ = dx_[i].ptr<float>(y);
            float* dy_row_ = dy_[i].ptr<float>(y);
            for (int x = 0; x < src[i].cols; ++x)
            {
                dx_row_[x] = normL2(dx_row[x]);
                dy_row_[x] = normL2(dy_row[x]);
            }
        }
    }
    PairwiseSeamFinder::find(src, corners, masks);
}


void GraphCutSeamFinderGpu::findInPair(size_t first, size_t second, Rect roi)
{
    Mat img1 = images_[first], img2 = images_[second];
    Mat dx1 = dx_[first], dx2 = dx_[second];
    Mat dy1 = dy_[first], dy2 = dy_[second];
    Mat mask1 = masks_[first], mask2 = masks_[second];
    Point tl1 = corners_[first], tl2 = corners_[second];

    const int gap = 10;
    Mat subimg1(roi.height + 2 * gap, roi.width + 2 * gap, CV_32FC3);
    Mat subimg2(roi.height + 2 * gap, roi.width + 2 * gap, CV_32FC3);
    Mat submask1(roi.height + 2 * gap, roi.width + 2 * gap, CV_8U);
    Mat submask2(roi.height + 2 * gap, roi.width + 2 * gap, CV_8U);
    Mat subdx1(roi.height + 2 * gap, roi.width + 2 * gap, CV_32F);
    Mat subdy1(roi.height + 2 * gap, roi.width + 2 * gap, CV_32F);
    Mat subdx2(roi.height + 2 * gap, roi.width + 2 * gap, CV_32F);
    Mat subdy2(roi.height + 2 * gap, roi.width + 2 * gap, CV_32F);

    // Cut subimages and submasks with some gap
    for (int y = -gap; y < roi.height + gap; ++y)
    {
        for (int x = -gap; x < roi.width + gap; ++x)
        {
            int y1 = roi.y - tl1.y + y;
            int x1 = roi.x - tl1.x + x;
            if (y1 >= 0 && x1 >= 0 && y1 < img1.rows && x1 < img1.cols)
            {
                subimg1.at<Point3f>(y + gap, x + gap) = img1.at<Point3f>(y1, x1);
                submask1.at<uchar>(y + gap, x + gap) = mask1.at<uchar>(y1, x1);
                subdx1.at<float>(y + gap, x + gap) = dx1.at<float>(y1, x1);
                subdy1.at<float>(y + gap, x + gap) = dy1.at<float>(y1, x1);
            }
            else
            {
                subimg1.at<Point3f>(y + gap, x + gap) = Point3f(0, 0, 0);
                submask1.at<uchar>(y + gap, x + gap) = 0;
                subdx1.at<float>(y + gap, x + gap) = 0.f;
                subdy1.at<float>(y + gap, x + gap) = 0.f;
            }

            int y2 = roi.y - tl2.y + y;
            int x2 = roi.x - tl2.x + x;
            if (y2 >= 0 && x2 >= 0 && y2 < img2.rows && x2 < img2.cols)
            {
                subimg2.at<Point3f>(y + gap, x + gap) = img2.at<Point3f>(y2, x2);
                submask2.at<uchar>(y + gap, x + gap) = mask2.at<uchar>(y2, x2);
                subdx2.at<float>(y + gap, x + gap) = dx2.at<float>(y2, x2);
                subdy2.at<float>(y + gap, x + gap) = dy2.at<float>(y2, x2);
            }
            else
            {
                subimg2.at<Point3f>(y + gap, x + gap) = Point3f(0, 0, 0);
                submask2.at<uchar>(y + gap, x + gap) = 0;
                subdx2.at<float>(y + gap, x + gap) = 0.f;
                subdy2.at<float>(y + gap, x + gap) = 0.f;
            }
        }
    }

    Mat terminals, leftT, rightT, top, bottom;

    switch (cost_type_)
    {
    case GraphCutSeamFinder::COST_COLOR:
        setGraphWeightsColor(subimg1, subimg2, submask1, submask2,
                             terminals, leftT, rightT, top, bottom);
        break;
    case GraphCutSeamFinder::COST_COLOR_GRAD:
        setGraphWeightsColorGrad(subimg1, subimg2, subdx1, subdx2, subdy1, subdy2,
                                 submask1, submask2, terminals, leftT, rightT, top, bottom);
        break;
    default:
        CV_Error(CV_StsBadArg, "unsupported pixel similarity measure");
    }

    gpu::GpuMat terminals_d(terminals);
    gpu::GpuMat leftT_d(leftT);
    gpu::GpuMat rightT_d(rightT);
    gpu::GpuMat top_d(top);
    gpu::GpuMat bottom_d(bottom);
    gpu::GpuMat labels_d, buf_d;

    gpu::graphcut(terminals_d, leftT_d, rightT_d, top_d, bottom_d, labels_d, buf_d);

    Mat_<uchar> labels = (Mat)labels_d;
    for (int y = 0; y < roi.height; ++y)
    {
        for (int x = 0; x < roi.width; ++x)
        {
            if (labels(y + gap, x + gap))
            {
                if (mask1.at<uchar>(roi.y - tl1.y + y, roi.x - tl1.x + x))
                    mask2.at<uchar>(roi.y - tl2.y + y, roi.x - tl2.x + x) = 0;
            }
            else
            {
                if (mask2.at<uchar>(roi.y - tl2.y + y, roi.x - tl2.x + x))
                    mask1.at<uchar>(roi.y - tl1.y + y, roi.x - tl1.x + x) = 0;
            }
        }
    }
}


void GraphCutSeamFinderGpu::setGraphWeightsColor(const Mat &img1, const Mat &img2, const Mat &mask1, const Mat &mask2,
                                                 Mat &terminals, Mat &leftT, Mat &rightT, Mat &top, Mat &bottom)
{
    const Size img_size = img1.size();

    terminals.create(img_size, CV_32S);
    leftT.create(Size(img_size.height, img_size.width), CV_32S);
    rightT.create(Size(img_size.height, img_size.width), CV_32S);
    top.create(img_size, CV_32S);
    bottom.create(img_size, CV_32S);

    Mat_<int> terminals_(terminals);
    Mat_<int> leftT_(leftT);
    Mat_<int> rightT_(rightT);
    Mat_<int> top_(top);
    Mat_<int> bottom_(bottom);

    // Set terminal weights
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            float source = mask1.at<uchar>(y, x) ? terminal_cost_ : 0.f;
            float sink = mask2.at<uchar>(y, x) ? terminal_cost_ : 0.f;
            terminals_(y, x) = saturate_cast<int>((source - sink) * 255.f);
        }
    }

    // Set regular edge weights
    const float weight_eps = 1.f;
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            if (x > 0)
            {
                float weight = normL2(img1.at<Point3f>(y, x - 1), img2.at<Point3f>(y, x - 1)) +
                               normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x)) +
                               weight_eps;
                if (!mask1.at<uchar>(y, x - 1) || !mask1.at<uchar>(y, x) ||
                    !mask2.at<uchar>(y, x - 1) || !mask2.at<uchar>(y, x))
                    weight += bad_region_penalty_;
                leftT_(x, y) = saturate_cast<int>(weight * 255.f);
            }
            else
                leftT_(x, y) = 0;

            if (x < img_size.width - 1)
            {
                float weight = normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x)) +
                               normL2(img1.at<Point3f>(y, x + 1), img2.at<Point3f>(y, x + 1)) +
                               weight_eps;
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y, x + 1) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y, x + 1))
                    weight += bad_region_penalty_;
                rightT_(x, y) = saturate_cast<int>(weight * 255.f);
            }
            else
                rightT_(x, y) = 0;

            if (y > 0)
            {
                float weight = normL2(img1.at<Point3f>(y - 1, x), img2.at<Point3f>(y - 1, x)) +
                               normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x)) +
                               weight_eps;
                if (!mask1.at<uchar>(y - 1, x) || !mask1.at<uchar>(y, x) ||
                    !mask2.at<uchar>(y - 1, x) || !mask2.at<uchar>(y, x))
                    weight += bad_region_penalty_;
                top_(y, x) = saturate_cast<int>(weight * 255.f);
            }
            else
                top_(y, x) = 0;

            if (y < img_size.height - 1)
            {
                float weight = normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x)) +
                               normL2(img1.at<Point3f>(y + 1, x), img2.at<Point3f>(y + 1, x)) +
                               weight_eps;
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y + 1, x) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y + 1, x))
                    weight += bad_region_penalty_;
                bottom_(y, x) = saturate_cast<int>(weight * 255.f);
            }
            else
                bottom_(y, x) = 0;
        }
    }
}


void GraphCutSeamFinderGpu::setGraphWeightsColorGrad(
        const Mat &img1, const Mat &img2, const Mat &dx1, const Mat &dx2,
        const Mat &dy1, const Mat &dy2, const Mat &mask1, const Mat &mask2,
        Mat &terminals, Mat &leftT, Mat &rightT, Mat &top, Mat &bottom)
{
    const Size img_size = img1.size();

    terminals.create(img_size, CV_32S);
    leftT.create(Size(img_size.height, img_size.width), CV_32S);
    rightT.create(Size(img_size.height, img_size.width), CV_32S);
    top.create(img_size, CV_32S);
    bottom.create(img_size, CV_32S);

    Mat_<int> terminals_(terminals);
    Mat_<int> leftT_(leftT);
    Mat_<int> rightT_(rightT);
    Mat_<int> top_(top);
    Mat_<int> bottom_(bottom);

    // Set terminal weights
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            float source = mask1.at<uchar>(y, x) ? terminal_cost_ : 0.f;
            float sink = mask2.at<uchar>(y, x) ? terminal_cost_ : 0.f;
            terminals_(y, x) = saturate_cast<int>((source - sink) * 255.f);
        }
    }

    // Set regular edge weights
    const float weight_eps = 1.f;
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            if (x > 0)
            {
                float grad = dx1.at<float>(y, x - 1) + dx1.at<float>(y, x) +
                             dx2.at<float>(y, x - 1) + dx2.at<float>(y, x) + weight_eps;
                float weight = (normL2(img1.at<Point3f>(y, x - 1), img2.at<Point3f>(y, x - 1)) +
                                normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x))) / grad +
                               weight_eps;
                if (!mask1.at<uchar>(y, x - 1) || !mask1.at<uchar>(y, x) ||
                    !mask2.at<uchar>(y, x - 1) || !mask2.at<uchar>(y, x))
                    weight += bad_region_penalty_;
                leftT_(x, y) = saturate_cast<int>(weight * 255.f);
            }
            else
                leftT_(x, y) = 0;

            if (x < img_size.width - 1)
            {
                float grad = dx1.at<float>(y, x) + dx1.at<float>(y, x + 1) +
                             dx2.at<float>(y, x) + dx2.at<float>(y, x + 1) + weight_eps;
                float weight = (normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x)) +
                                normL2(img1.at<Point3f>(y, x + 1), img2.at<Point3f>(y, x + 1))) / grad +
                               weight_eps;
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y, x + 1) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y, x + 1))
                    weight += bad_region_penalty_;
                rightT_(x, y) = saturate_cast<int>(weight * 255.f);
            }
            else
                rightT_(x, y) = 0;

            if (y > 0)
            {
                float grad = dy1.at<float>(y - 1, x) + dy1.at<float>(y, x) +
                             dy2.at<float>(y - 1, x) + dy2.at<float>(y, x) + weight_eps;
                float weight = (normL2(img1.at<Point3f>(y - 1, x), img2.at<Point3f>(y - 1, x)) +
                                normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x))) / grad +
                               weight_eps;
                if (!mask1.at<uchar>(y - 1, x) || !mask1.at<uchar>(y, x) ||
                    !mask2.at<uchar>(y - 1, x) || !mask2.at<uchar>(y, x))
                    weight += bad_region_penalty_;
                top_(y, x) = saturate_cast<int>(weight * 255.f);
            }
            else
                top_(y, x) = 0;

            if (y < img_size.height - 1)
            {
                float grad = dy1.at<float>(y, x) + dy1.at<float>(y + 1, x) +
                             dy2.at<float>(y, x) + dy2.at<float>(y + 1, x) + weight_eps;
                float weight = (normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x)) +
                                normL2(img1.at<Point3f>(y + 1, x), img2.at<Point3f>(y + 1, x))) / grad +
                               weight_eps;
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y + 1, x) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y + 1, x))
                    weight += bad_region_penalty_;
                bottom_(y, x) = saturate_cast<int>(weight * 255.f);
            }
            else
                bottom_(y, x) = 0;
        }
    }
}
#endif
*/
