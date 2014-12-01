
/*LSH's alteration based on OpenCV 2.4.6*/

#include "KRwarper.h"


/****************************************** warp Part ************************************************/

void WarperProjector::setCameraParams(const Mat &K, const Mat &R, const Mat &T)
{
	CV_Assert(K.size() == Size(3, 3) && K.type() == CV_32F);
	CV_Assert(R.size() == Size(3, 3) && R.type() == CV_32F);
	CV_Assert((T.size() == Size(1, 3) || T.size() == Size(3, 1)) && T.type() == CV_32F);

	Mat_<float> K_(K);
	k[0] = K_(0,0); k[1] = K_(0,1); k[2] = K_(0,2);
	k[3] = K_(1,0); k[4] = K_(1,1); k[5] = K_(1,2);
	k[6] = K_(2,0); k[7] = K_(2,1); k[8] = K_(2,2);

	Mat_<float> Rinv = R.t();
	rinv[0] = Rinv(0,0); rinv[1] = Rinv(0,1); rinv[2] = Rinv(0,2);
	rinv[3] = Rinv(1,0); rinv[4] = Rinv(1,1); rinv[5] = Rinv(1,2);
	rinv[6] = Rinv(2,0); rinv[7] = Rinv(2,1); rinv[8] = Rinv(2,2);

	Mat_<float> R_Kinv = R * K.inv();
	r_kinv[0] = R_Kinv(0,0); r_kinv[1] = R_Kinv(0,1); r_kinv[2] = R_Kinv(0,2);
	r_kinv[3] = R_Kinv(1,0); r_kinv[4] = R_Kinv(1,1); r_kinv[5] = R_Kinv(1,2);
	r_kinv[6] = R_Kinv(2,0); r_kinv[7] = R_Kinv(2,1); r_kinv[8] = R_Kinv(2,2);

	Mat_<float> K_Rinv = K * Rinv;
	k_rinv[0] = K_Rinv(0,0); k_rinv[1] = K_Rinv(0,1); k_rinv[2] = K_Rinv(0,2);
	k_rinv[3] = K_Rinv(1,0); k_rinv[4] = K_Rinv(1,1); k_rinv[5] = K_Rinv(1,2);
	k_rinv[6] = K_Rinv(2,0); k_rinv[7] = K_Rinv(2,1); k_rinv[8] = K_Rinv(2,2);

	Mat_<float> T_(T.reshape(0, 3));
	t[0] = T_(0,0); t[1] = T_(1,0); t[2] = T_(2,0);
}

//����ͶӰ
inline
void CylindricalProjector::mapForward(float x, float y, float &u, float &v)
{
	float x_ = r_kinv[0] * x + r_kinv[1] * y + r_kinv[2];
	float y_ = r_kinv[3] * x + r_kinv[4] * y + r_kinv[5];
	float z_ = r_kinv[6] * x + r_kinv[7] * y + r_kinv[8];

	u = scale * atan2f(x_, z_);
	v = scale * y_ / sqrtf(x_ * x_ + z_ * z_);
}

inline
void CylindricalProjector::mapBackward(float u, float v, float &x, float &y)
{
	u /= scale;
	v /= scale;

	float x_ = sinf(u);
	float y_ = v;
	float z_ = cosf(u);

	float z;
	x = k_rinv[0] * x_ + k_rinv[1] * y_ + k_rinv[2] * z_;
	y = k_rinv[3] * x_ + k_rinv[4] * y_ + k_rinv[5] * z_;
	z = k_rinv[6] * x_ + k_rinv[7] * y_ + k_rinv[8] * z_;

	if (z > 0) { x /= z; y /= z; }
	else x = y = -1;
}

//ƽ��ͶӰ
inline
void PlaneProjector::mapForward(float x, float y, float &u, float &v)
{
	float x_ = r_kinv[0] * x + r_kinv[1] * y + r_kinv[2];
	float y_ = r_kinv[3] * x + r_kinv[4] * y + r_kinv[5];
	float z_ = r_kinv[6] * x + r_kinv[7] * y + r_kinv[8];

	x_ = t[0] + x_ / z_ * (1 - t[2]);
	y_ = t[1] + y_ / z_ * (1 - t[2]);

	u = scale * x_;
	v = scale * y_;
}

inline
void PlaneProjector::mapBackward(float u, float v, float &x, float &y)
{
	u = u / scale - t[0];
	v = v / scale - t[1];

	float z;
	x = k_rinv[0] * u + k_rinv[1] * v + k_rinv[2] * (1 - t[2]);
	y = k_rinv[3] * u + k_rinv[4] * v + k_rinv[5] * (1 - t[2]);
	z = k_rinv[6] * u + k_rinv[7] * v + k_rinv[8] * (1 - t[2]);

	x /= z;
	y /= z;
}

//����ͶӰ
inline
void SphericalProjector::mapForward(float x, float y, float &u, float &v)
{
	float x_ = r_kinv[0] * x + r_kinv[1] * y + r_kinv[2];
	float y_ = r_kinv[3] * x + r_kinv[4] * y + r_kinv[5];
	float z_ = r_kinv[6] * x + r_kinv[7] * y + r_kinv[8];

	u = scale * atan2f(x_, z_);
	float w = y_ / sqrtf(x_ * x_ + y_ * y_ + z_ * z_);
	v = scale * (static_cast<float>(CV_PI) - acosf(w == w ? w : 0));
}

inline
void SphericalProjector::mapBackward(float u, float v, float &x, float &y)
{
	u /= scale;
	v /= scale;

	float sinv = sinf(static_cast<float>(CV_PI) - v);
	float x_ = sinv * sinf(u);
	float y_ = cosf(static_cast<float>(CV_PI) - v);
	float z_ = sinv * cosf(u);

	float z;
	x = k_rinv[0] * x_ + k_rinv[1] * y_ + k_rinv[2] * z_;
	y = k_rinv[3] * x_ + k_rinv[4] * y_ + k_rinv[5] * z_;
	z = k_rinv[6] * x_ + k_rinv[7] * y_ + k_rinv[8] * z_;

	if (z > 0) { x /= z; y /= z; }
	else x = y = -1;
}



template <class P>
Point2f KRWarperbase<P>::warpPoint(const Point2f &pt, const Mat &K, const Mat &R)
{
	projector_.setCameraParams(K, R);
	Point2f uv;
	projector_.mapForward(pt.x, pt.y, uv.x, uv.y);
	return uv;
}


template <class P>
Rect KRWarperbase<P>::buildMaps(Size src_size, const Mat &K, const Mat &R, Mat &xmap, Mat &ymap)
{
	projector_.setCameraParams(K, R);

	Point dst_tl, dst_br;
	detectResultRoi(src_size, dst_tl, dst_br);

	xmap.create(dst_br.y - dst_tl.y + 1, dst_br.x - dst_tl.x + 1, CV_32F);
	ymap.create(dst_br.y - dst_tl.y + 1, dst_br.x - dst_tl.x + 1, CV_32F);

	float x, y;
	for (int v = dst_tl.y; v <= dst_br.y; ++v)
	{
		for (int u = dst_tl.x; u <= dst_br.x; ++u)
		{
			projector_.mapBackward(static_cast<float>(u), static_cast<float>(v), x, y);
			xmap.at<float>(v - dst_tl.y, u - dst_tl.x) = x;
			ymap.at<float>(v - dst_tl.y, u - dst_tl.x) = y;
		}
	}

	return Rect(dst_tl, dst_br);
}


template <class P>
Point KRWarperbase<P>::warp(const Mat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
								  Mat &dst)
{
	Mat xmap, ymap;
	Rect dst_roi = buildMaps(src.size(), K, R, xmap, ymap);

	dst.create(dst_roi.height + 1, dst_roi.width + 1, src.type());
	remap(src, dst, xmap, ymap, interp_mode, border_mode);

	return dst_roi.tl();
}

template <class P>
Rect KRWarperbase<P>::prepare( const Mat &src, const Mat &K, const Mat &R, Mat &xmap, Mat &ymap)
{
	Rect dst_roi = buildMaps(src.size(), K, R, xmap, ymap);
	return dst_roi;
}

template <class P>
void KRWarperbase<P>::doWarp( const Mat &src, Mat &xmap, Mat &ymap, int interp_mode, int border_mode, Mat &dst)
{
	remap(src,dst, xmap, ymap, interp_mode, border_mode);
}


template <class P>
void KRWarperbase<P>::warpBackward(const Mat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
										 Size dst_size, Mat &dst)
{
	projector_.setCameraParams(K, R);

	Point src_tl, src_br;
	detectResultRoi(dst_size, src_tl, src_br);
	CV_Assert(src_br.x - src_tl.x + 1 == src.cols && src_br.y - src_tl.y + 1 == src.rows);

	Mat xmap(dst_size, CV_32F);
	Mat ymap(dst_size, CV_32F);

	float u, v;
	for (int y = 0; y < dst_size.height; ++y)
	{
		for (int x = 0; x < dst_size.width; ++x)
		{
			projector_.mapForward(static_cast<float>(x), static_cast<float>(y), u, v);
			xmap.at<float>(y, x) = u - src_tl.x;
			ymap.at<float>(y, x) = v - src_tl.y;
		}
	}

	dst.create(dst_size, src.type());
	remap(src, dst, xmap, ymap, interp_mode, border_mode);
}


template <class P>
Rect KRWarperbase<P>::warpRoi(Size src_size, const Mat &K, const Mat &R)
{
	projector_.setCameraParams(K, R);

	Point dst_tl, dst_br;
	detectResultRoi(src_size, dst_tl, dst_br);

	return Rect(dst_tl, Point(dst_br.x + 1, dst_br.y + 1));
}


template <class P>
void KRWarperbase<P>::detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br)
{
	float tl_uf = std::numeric_limits<float>::max();
	float tl_vf = std::numeric_limits<float>::max();
	float br_uf = -std::numeric_limits<float>::max();
	float br_vf = -std::numeric_limits<float>::max();

	float u, v;
	for (int y = 0; y < src_size.height; ++y)
	{
		for (int x = 0; x < src_size.width; ++x)
		{
			projector_.mapForward(static_cast<float>(x), static_cast<float>(y), u, v);
			tl_uf = std::min(tl_uf, u); tl_vf = std::min(tl_vf, v);
			br_uf = std::max(br_uf, u); br_vf = std::max(br_vf, v);
		}
	}

	dst_tl.x = static_cast<int>(tl_uf);
	dst_tl.y = static_cast<int>(tl_vf);
	dst_br.x = static_cast<int>(br_uf);
	dst_br.y = static_cast<int>(br_vf);
}


template <class P>
void KRWarperbase<P>::detectResultRoiByBorder(Size src_size, Point &dst_tl, Point &dst_br)
{
	float tl_uf = std::numeric_limits<float>::max();
	float tl_vf = std::numeric_limits<float>::max();
	float br_uf = -std::numeric_limits<float>::max();
	float br_vf = -std::numeric_limits<float>::max();

	float u, v;
	for (float x = 0; x < src_size.width; ++x)
	{
		projector_.mapForward(static_cast<float>(x), 0, u, v);
		tl_uf = std::min(tl_uf, u); tl_vf = std::min(tl_vf, v);
		br_uf = std::max(br_uf, u); br_vf = std::max(br_vf, v);

		projector_.mapForward(static_cast<float>(x), static_cast<float>(src_size.height - 1), u, v);
		tl_uf = std::min(tl_uf, u); tl_vf = std::min(tl_vf, v);
		br_uf = std::max(br_uf, u); br_vf = std::max(br_vf, v);
	}
	for (int y = 0; y < src_size.height; ++y)
	{
		projector_.mapForward(0, static_cast<float>(y), u, v);
		tl_uf = std::min(tl_uf, u); tl_vf = std::min(tl_vf, v);
		br_uf = std::max(br_uf, u); br_vf = std::max(br_vf, v);

		projector_.mapForward(static_cast<float>(src_size.width - 1), static_cast<float>(y), u, v);
		tl_uf = std::min(tl_uf, u); tl_vf = std::min(tl_vf, v);
		br_uf = std::max(br_uf, u); br_vf = std::max(br_vf, v);
	}

	dst_tl.x = static_cast<int>(tl_uf);
	dst_tl.y = static_cast<int>(tl_vf);
	dst_br.x = static_cast<int>(br_uf);
	dst_br.y = static_cast<int>(br_vf);
}


void PlaneWarper::detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br)
{
	float tl_uf = numeric_limits<float>::max();
	float tl_vf = numeric_limits<float>::max();
	float br_uf = -numeric_limits<float>::max();
	float br_vf = -numeric_limits<float>::max();

	float u, v;

	projector_.mapForward(0, 0, u, v);
	tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
	br_uf = max(br_uf, u); br_vf = max(br_vf, v);

	projector_.mapForward(0, static_cast<float>(src_size.height - 1), u, v);
	tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
	br_uf = max(br_uf, u); br_vf = max(br_vf, v);

	projector_.mapForward(static_cast<float>(src_size.width - 1), 0, u, v);
	tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
	br_uf = max(br_uf, u); br_vf = max(br_vf, v);

	projector_.mapForward(static_cast<float>(src_size.width - 1), static_cast<float>(src_size.height - 1), u, v);
	tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
	br_uf = max(br_uf, u); br_vf = max(br_vf, v);

	dst_tl.x = static_cast<int>(tl_uf);
	dst_tl.y = static_cast<int>(tl_vf);
	dst_br.x = static_cast<int>(br_uf);
	dst_br.y = static_cast<int>(br_vf);
}


void SphericalWarper::detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br)
{
	detectResultRoiByBorder(src_size, dst_tl, dst_br);

	float tl_uf = static_cast<float>(dst_tl.x);
	float tl_vf = static_cast<float>(dst_tl.y);
	float br_uf = static_cast<float>(dst_br.x);
	float br_vf = static_cast<float>(dst_br.y);

	float x = projector_.rinv[1];
	float y = projector_.rinv[4];
	float z = projector_.rinv[7];
	if (y > 0.f)
	{
		float x_ = (projector_.k[0] * x + projector_.k[1] * y) / z + projector_.k[2];
		float y_ = projector_.k[4] * y / z + projector_.k[5];
		if (x_ > 0.f && x_ < src_size.width && y_ > 0.f && y_ < src_size.height)
		{
			tl_uf = min(tl_uf, 0.f); tl_vf = min(tl_vf, static_cast<float>(CV_PI * projector_.scale));
			br_uf = max(br_uf, 0.f); br_vf = max(br_vf, static_cast<float>(CV_PI * projector_.scale));
		}
	}

	x = projector_.rinv[1];
	y = -projector_.rinv[4];
	z = projector_.rinv[7];
	if (y > 0.f)
	{
		float x_ = (projector_.k[0] * x + projector_.k[1] * y) / z + projector_.k[2];
		float y_ = projector_.k[4] * y / z + projector_.k[5];
		if (x_ > 0.f && x_ < src_size.width && y_ > 0.f && y_ < src_size.height)
		{
			tl_uf = min(tl_uf, 0.f); tl_vf = min(tl_vf, static_cast<float>(0));
			br_uf = max(br_uf, 0.f); br_vf = max(br_vf, static_cast<float>(0));
		}
	}

	dst_tl.x = static_cast<int>(tl_uf);
	dst_tl.y = static_cast<int>(tl_vf);
	dst_br.x = static_cast<int>(br_uf);
	dst_br.y = static_cast<int>(br_vf);
}

template class KRWarperbase<CylindricalProjector>;
template class KRWarperbase<SphericalProjector>;
template class KRWarperbase<PlaneProjector>;



