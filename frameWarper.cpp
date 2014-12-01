#include "frameWarper.h"

frameWarper::frameWarper()
{
}


double getBestWarperSize(vector<CameraParams> &cameras)//计算warper的合适尺度
{
	int video_num=(int)cameras.size();
	vector<double> focals;
	focals.resize(video_num);

	for (int i=0; i<video_num;i++)
	{
		focals[i] = cameras[i].focal;
	}

	sort(focals.begin(), focals.end());
	if(focals.size() % 2 == 1)
		return focals[focals.size()/2];
	else
		return ((focals[focals.size()/2] + focals[focals.size()/2-1])*0.5f);
}


template<class w>
void KRBasedWarp<w>::estimate(vector<Mat> &imgSet, vector<imgFeatures> &imgF,
							  vector<mLog> &matchInfoIn,
							  vector<CameraParams> &cameras)
{

	bool is_estimate_finish = kr_estimator.estimate(imgSet,imgF,matchInfoIn,cameras);

	unsigned int video_num = kr_estimator.matcher.videosFlag.size();

	if(!is_estimate_finish)
		throw sysException("Error: Parameters estimation failed!");

	float size = (float)getBestWarperSize(cameras);

	warper.setScale(size);

}

template<class w>
void KRBasedWarp<w>::prepare(vector<Mat> &imgSet, vector<imgFeatures> &imgF, vector<mLog> &matchInfoIn, 
							 vector<Mat> &imgMaskWarpOut,
							 vector<Point> &topleftOut,
							 vector<bool> &videoFlag)
{

	vector<CameraParams> cameras;
	unsigned int origin_num = (unsigned int)imgF.size();//原始视频数目

	estimate(imgSet, imgF, matchInfoIn, cameras);

	unsigned int video_num = (unsigned int)imgSet.size();//估计和剪枝后剩余视频数目

	//保存筛选结果videoFlag
	videoFlag.clear();
	videoFlag.resize(origin_num);
	int sum = 0;
	for(unsigned int i=0;i<origin_num;i++)
	{
		if (true==kr_estimator.matcher.videosFlag[i])
		{
			videoFlag[i]=true;
			sum++;
		}
		else
			videoFlag[i]=false;
	}
	
	//判断筛选视频结果是否和videoFlag信息一致 question by say:为啥会不一致呢？
	if(sum!=(int)video_num)
	{
		throw sysException("[Failed] K-R estimation failed: non-consistent number of flag and selected videos!");
	}

	cameras.resize(video_num);
	topleftOut.resize(video_num);
	xmap_list.resize(video_num);
	ymap_list.resize(video_num);
	imgMaskWarpOut.resize(video_num);
	Mat mask;
	Rect dst_roi;
	for(unsigned int i=0;i<video_num;i++)
	{
		mask.create(imgSet[i].rows,imgSet[i].cols, CV_8UC1);
		mask.setTo(Scalar::all(255));
		dst_roi = prepare(imgSet[i],cameras[i], xmap_list[i], ymap_list[i]);
		topleftOut[i] = dst_roi.tl();
		imgMaskWarpOut[i].create(dst_roi.height + 1, dst_roi.width + 1, mask.type());
		warp(mask, xmap_list[i], ymap_list[i],INTER_NEAREST, BORDER_CONSTANT, imgMaskWarpOut[i]);
	}
}

template<class w>
Rect KRBasedWarp<w>::prepare(const Mat &src, CameraParams &camera, Mat &xmap, Mat &ymap)
{

	Mat K;
	camera.K().convertTo(K,CV_32F);
	Rect dst_roi = warper.prepare(src, K, camera.R, xmap, ymap);
	return dst_roi;

}

template<class w>
void KRBasedWarp<w>::warp(const Mat &src, Mat &xmap, Mat &ymap, int interp_mode, int border_mode,Mat &dst)
{
	warper.doWarp(src,xmap,ymap,interp_mode,border_mode,dst);
}

template<class w>
void KRBasedWarp<w>::doWarp(vector<Mat> &imgSet,vector<Mat>&imgWarpOut)
{
	imgWarpOut.resize(imgSet.size());

	for(unsigned int i=0;i<imgSet.size();i++)
	{
      warp(imgSet[i], xmap_list[i], ymap_list[i],INTER_NEAREST, BORDER_REFLECT, imgWarpOut[i]);
	}
}

template class KRBasedWarp<CylindricalWarper>;
template class KRBasedWarp<SphericalWarper>;
template class KRBasedWarp<PlaneWarper>;
