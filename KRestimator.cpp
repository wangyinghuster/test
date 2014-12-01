#include "KRestimator.h"

/********************************* match points further process***************************************/

int OptimizeMatch::expandMatchInfo(vector<Mat> &imgSet, vector<imgFeatures> &imgF,vector<mLog> &matchIdx, homoT *root)
{
	unsigned int videoNums = (unsigned int)imgSet.size();
	videosFlag.resize(videoNums);
	match_info.resize( matchIdx.size());
	calAllMatch(imgSet,imgF,matchIdx);
	
	// 构建匹配树
		
	__debug(
			cout << "[Info]Generate match tree from match set..." << endl;
		)
	
	//构建最大匹配关系树，并调整匹配关系的query和train
	MaxSpanningTree(root,videoNums); 

	adjustVideos(imgSet);

	//调整索引值
 	return adjustIndx( root, videoNums);

}




//计算变换矩阵和匹配关系,主要是query转换到train图像的转换矩阵H生成
int OptimizeMatch::calAllMatch(vector<Mat> &imgSet, vector<imgFeatures> &imgF,vector<mLog> &matchIdx)
{
	vector<Point2f> src;//transform
	vector<Point2f> dst;//base
	vector<DMatch> matchD;
	vector<KeyPoint> keyT,keyB;

	unsigned int match_num =(unsigned int) matchIdx.size();

	for(unsigned int id=0; id<match_num; ++id)
	{
		match_info[id].queryInx = matchIdx[id].queryInx;
		match_info[id].trainInx = matchIdx[id].trainInx;
		match_info[id].matchPointIndex = matchIdx[id].matchPointIndex;
		match_info[id].matchPointIndexRev = matchIdx[id].matchPointIndexRev;
		matchD = matchIdx[id].matchPointIndex;
		src.resize(matchD.size());
		dst.resize(matchD.size());

		keyT = imgF[matchIdx[id].queryInx].backGroundPoint;//query——src
		keyB = imgF[matchIdx[id].trainInx].backGroundPoint;//train——dst

		float width_qI  = imgSet[matchIdx[id].queryInx].cols*0.5f;//这四行提到下个for循环的前面来，以减少运算量
		float height_qI = imgSet[matchIdx[id].queryInx].rows*0.5f;

		float width_tI  = imgSet[matchIdx[id].trainInx].cols*0.5f;
		float height_tI = imgSet[matchIdx[id].trainInx].rows*0.5f;

		for (unsigned int i=0;i<matchD.size();i++)
		{
			Point2f p = keyT[matchD[i].queryIdx].pt;
			p.x -= width_qI;
			p.y -= height_qI;
			src[i] = p;
			//src.push_back(keyT[matchD[i].queryIdx].pt);
			p = keyB[matchD[i].trainIdx].pt;
			p.x -= width_tI;
			p.y -= height_tI;
			dst[i] = p;
			//dst.push_back(keyB[matchD[i].trainIdx].pt);
		}

		match_info[id].H = findHomography(src,dst,match_info[id].inliers_mask,CV_RANSAC,3.0); //默认阈为3.0
		match_info[id].num_inliers = 0;
		for (size_t z = 0; z < match_info[id].inliers_mask.size(); ++z)
		{
			if (match_info[id].inliers_mask[z])
				match_info[id].num_inliers++;
		}

		//计算反变换
		match_info[id].HRev= findHomography(dst,src,match_info[id].inliers_mask_rev,CV_RANSAC,3.0); //默认阈为3.0
		match_info[id].num_inliers_rev = 0;
		for(size_t z = 0; z < match_info[id].inliers_mask_rev.size(); ++z)
		{
			if (match_info[id].inliers_mask_rev[z])
				match_info[id].num_inliers_rev++;
		}

	}

	return 1;
}

int OptimizeMatch::deleteTree(homoT *root)
{
	homoT *nowPoint,*temp;
	nowPoint=root->son;
	while (nowPoint!=NULL){
		temp=nowPoint->next;
		deleteTree(nowPoint);
		nowPoint=temp;
	}
	delete root;
	return 1;
}

int OptimizeMatch::printTree(homoT *root)
{
	homoT *nowPoint;

	cout <<"[Info]root "<< root->picInx<<":";

	nowPoint=root->son;
	while (nowPoint!=NULL){
		cout << nowPoint->picInx<<"  ";
		nowPoint=nowPoint->next;
	}
	cout << endl;
	nowPoint=root->son;
	while (nowPoint!=NULL){
		printTree(nowPoint);
		nowPoint=nowPoint->next;
	}
	return 1;
}


void OptimizeMatch::MaxSpanningTree( homoT *homoTRoot, unsigned int video_num)
{
	vector<mLog>::size_type matchSize = match_info.size();
	int *imgInd = new int[video_num];

/*
	//筛选置信度超过一定阈值的匹配关系
	vector<exmLog> matVecTemp;
	for (unsigned int i=0;i< match_info.size();i++)
	{
		__debug(
			cout << "[Info]Finding match between img "<< match_info[i].queryInx
			<<" and img "<< match_info[i].trainInx
			<<", C="<< match_info[i].confidenc<<" ("<< match_info[i].num_inliers <<")"
			<<", C_R="<< match_info[i].confidencRev<<" ("<< match_info[i].num_inliers_rev <<")...";	
		)
			if (( match_info[i].confidenc>conf_thrd)&&( match_info[i].confidencRev>conf_thrd))
			{
				matVecTemp.push_back(match_info[i]);

				__debug(cout <<"Accepted"<< endl;)

			}
			__debug(
			else
			cout <<"Rejected"<< endl;
		)
	}
	match_info = matVecTemp;*/


    //最大生成树算法，采用Kruskal算法
    sort(match_info.begin(),match_info.end());  //由于重载了operator ＜ 升序排序,权值为匹配点的数目
    int *setCount=new int[video_num];
    for (int k=0;k<video_num;k++) 
		setCount[k]=0;


    int setFlag=1;

    for (int k=(match_info.size()-1);k>=0;k--)
    {
        if ((setCount[match_info[k].queryInx]==0)&&(setCount[match_info[k].trainInx]==0))
		{   //两个点都没有加入生成集合，则这两个点加入一个新集合
            setCount[match_info[k].queryInx]=setFlag;
            setCount[match_info[k].trainInx]=setFlag;
            setFlag++;
            match_info[k].inTree=true;  //这条边加入集合
			match_info[k].used = false;
        }else if ((setCount[match_info[k].queryInx]==0)&&(setCount[match_info[k].trainInx]!=0))
		{ //其中一个点没加入集合，则加入
            setCount[match_info[k].queryInx]=setCount[match_info[k].trainInx];
            match_info[k].inTree=true;  //这条边加入集合
			match_info[k].used = false;
        }
		else if ((setCount[match_info[k].queryInx]!=0)&&(setCount[match_info[k].trainInx]==0))
		{ //其中一个点没加入集合，则加入
            setCount[match_info[k].trainInx]=setCount[match_info[k].queryInx];
            match_info[k].inTree=true;  //这条边加入集合
			match_info[k].used = false;
        }
		else if (setCount[match_info[k].queryInx]!=setCount[match_info[k].trainInx])
		{   //两个点加入了不同集合，则连接这两个集合，将集合代码统一化
            match_info[k].inTree=true;  //这条边加入集合
			match_info[k].used=false;
            int needChange=setCount[match_info[k].trainInx];
            int chage=setCount[match_info[k].queryInx];
            for (int w=0;w<video_num;w++){
                if (setCount[w]==needChange){
                    setCount[w]=chage;
                }
            } //更新集合的代码
        } //其余的情况，不予增加
		match_info[k].used=false;
    }
    delete [] setCount;

/*	// 寻找匹配点数目最多的图片
	//这边留一个想法，如果采用寻找使得树最矮的点是不是可以更好？绝对是更好~所以赶紧想
	for (unsigned int i=0;i<video_num;i++){
		imgInd[i]=0;
	}

	for (vector<mLog>::size_type i=0;i<match_info.size();i++){
		imgInd[match_info[i].queryInx]+=(int)(match_info[i].matchPointIndex.size());
		imgInd[match_info[i].trainInx]+=(int)(match_info[i].matchPointIndex.size());
	}

	int baseImgFact=0;
	for (unsigned int i=0;i<video_num;i++){
		if (imgInd[i]>imgInd[baseImgFact]){
			baseImgFact=i;
		}
	}
*/
	//下面来实现这个算法：
	//将连通无向无环图转换成深度最小的树的代码：
	//每次删除所有的叶节点，直到只剩下一个或者两个节点为止

	for (unsigned int i=0;i<video_num;i++){
		imgInd[i]=0;
	}
	int cnt=video_num;
	while (cnt>2){
		for (vector<mLog>::size_type i=0;i<match_info.size();i++){
			if ((match_info[i].inTree)&&(imgInd[match_info[i].queryInx]!=-1)&&(imgInd[match_info[i].trainInx]!=-1)){
				imgInd[match_info[i].queryInx]++;
				imgInd[match_info[i].trainInx]++;
			}
		}
		cnt = 0;
		for (unsigned int i=0;i<video_num;i++){
			if (imgInd[i]==1){
				imgInd[i]=-1;
			}else if (imgInd[i]!=-1){
				imgInd[i]=0;
				cnt++;
			}
		}
	}

	int baseImgFact=0;
	while(imgInd[baseImgFact]==-1) baseImgFact++;

	// 将这个图片的平面作为主平面
	__debug(
		cout << "[Info]Setting video input "<< baseImgFact <<" as base img..." << endl;
	)

	homoTRoot->picInx=baseImgFact;
	homoTRoot->next=NULL;
	videosFlag[baseImgFact]=true;

	delete [] imgInd;

	// 构建匹配树
	__debug(
		cout << "[Info]Building match tree from match set..." << endl;
	)
		buildHomoTree(homoTRoot); //构建最大匹配关系树，并调整匹配关系的query和train
    __debug(printTree(homoTRoot);)

	__debug(
	cout << "[Info]Adjusting match index and videos index..." << endl;
	)

}


//以找到的基准平面出发构建最大匹配树2014.2.20
//同时调整保存匹配信息的向量matchIdx
//使得所有的nowPoint->picInx =  matchIdx[nowPoint->matId].query
int OptimizeMatch::buildHomoTree( homoT *root)
{
	int rootId=root->picInx;
	int sonNum=0;
	homoT *nowPoint= NULL;
	//初始条件
	root->son=nowPoint;
	unsigned int matchSize=(unsigned int)match_info.size();

	for (unsigned int i=0;i<matchSize;i++)
	{
		if ((match_info[i].inTree==true)&&(match_info[i].used==false))   //这里只对存在于树的边进行操作
		{
			if ((match_info[i].queryInx == rootId)||(match_info[i].trainInx == rootId))
			{
				if (match_info[i].queryInx == rootId)
				{   //如果queryInx是父节点，那么交换这个matchIdx的东西。
					exmLog matTemp;
					matTemp.trainInx = match_info[i].queryInx;
					matTemp.queryInx = match_info[i].trainInx;
					matTemp.matchPointIndex = match_info[i].matchPointIndexRev;
					matTemp.matchPointIndexRev =match_info[i].matchPointIndex;
					matTemp.H =match_info[i].HRev;  //这边不用clone，智能指针会自动选择是否释放空间
					matTemp.HRev =match_info[i].H;

					matTemp.inliers_mask=match_info[i].inliers_mask_rev;
					matTemp.inliers_mask_rev=match_info[i].inliers_mask;
					matTemp.num_inliers=match_info[i].num_inliers_rev;
					matTemp.num_inliers_rev=match_info[i].num_inliers;

					match_info[i]=matTemp;
				}			
				if (sonNum==0)
				{
					nowPoint=new homoT;
					root->son=nowPoint;
					sonNum++;
				}
				else					
				{
					nowPoint->next=new homoT;
					nowPoint=nowPoint->next;
					sonNum++;
				}
				
				nowPoint->picInx=match_info[i].queryInx;
				videosFlag[match_info[i].queryInx]=true;
				nowPoint->matId=i;
				nowPoint->next=NULL;
				match_info[i].used=true;
			}
		
			//videosFlag[match_info[i].queryInx]=false;
			
		}	
	}

	nowPoint=root->son;
	while (nowPoint!=NULL){
		buildHomoTree(nowPoint);
		nowPoint=nowPoint->next;
	}
	return 1;
}


//videoAdjust会被调整，并缩小
int OptimizeMatch::adjustVideos(vector<Mat> &videoAdjust)
{
	vector<Mat> videosNew;

	for (unsigned int i=0;i<videoAdjust.size();i++)
	{
		if (videosFlag[i]==true)
		{  //如果这个视频在树中
			videosNew.push_back(videoAdjust[i]);
		}
	}

	videoAdjust=videosNew;
	return 1;
}

//调整索引值，从videoSeq中删去没有在树中的videos，返回值为缩减之后的videos的个数。
int OptimizeMatch::adjustIndx(homoT *homoTRoot, unsigned int videoSize)
{
	int *indxMap=new int[videoSize];

	int inx=0;
	for (unsigned int i=0;i<videoSize;i++)
	{
		if (videosFlag[i]==true)
		{  //如果这个视频在树中
			indxMap[i]=inx;
			inx++;
		}else{
			indxMap[i]=-1;
		}
	}

	int ans=inx-1;

	for (unsigned int i=0;i<match_info.size();i++){    //更新mlog里面的索引
		match_info[i].queryInx=indxMap[match_info[i].queryInx];
		match_info[i].trainInx=indxMap[match_info[i].trainInx];
	}

	int *mlogInxMap=new int[match_info.size()];  //对mlog向量删减,删减那些和树没有任何一个节点相连接的匹配关系

	vector<exmLog> newMathcIdx;
	inx=0;
	for (unsigned int i=0;i<match_info.size();i++){
		if ((match_info[i].queryInx>=0)&&(match_info[i].trainInx>=0)){  //如果没有连接
			newMathcIdx.push_back(match_info[i]);
			mlogInxMap[i]=inx;
			inx++;
		}else{
			mlogInxMap[i]=-1;
		}
	}

	match_info=newMathcIdx;
	newMathcIdx.clear();

	//对树的idx和matId进行更新

	homoTRoot->picInx=indxMap[homoTRoot->picInx];

	updateTreIdx(homoTRoot,indxMap,mlogInxMap);

	delete [] indxMap;   //释放空间
	delete [] mlogInxMap;

	return ans;
}

int OptimizeMatch::updateTreIdx(homoT *root, int idxMap[], int matIdxMap[])
{
	homoT *nowPoint;

	nowPoint=root->son;
	while (nowPoint!=NULL)
	{
		nowPoint->matId=matIdxMap[nowPoint->matId];  //按照映射关系更新索引
		nowPoint->picInx=idxMap[nowPoint->picInx];

		nowPoint=nowPoint->next;
	}

	nowPoint=root->son;
	while (nowPoint!=NULL){
		updateTreIdx(nowPoint,idxMap,matIdxMap);
		nowPoint=nowPoint->next;
	}
	return 1;

}



/********************************* parameter estimation part******************************************/

void calcDeriv(const Mat &err1, const Mat &err2, double h, Mat res)
{
	for (int i = 0; i < err1.rows; ++i)
		res.at<double>(i, 0) = (err2.at<double>(i, 0) - err1.at<double>(i, 0)) / h;
}


void KREstimator::estimateFocal(vector<Mat> &imgSet,vector<exmLog> &match_info,vector<double> &focals)
{
	int num_videos = imgSet.size();
	focals.resize(num_videos);
	vector<double> all_focals;
	for (size_t i=0;i<match_info.size();++i)
	{
		double f0, f1;
		bool  f0_OK, f1_OK;
		focalsFromHomography(match_info[i].H, f0, f1, f0_OK, f1_OK);
		if ( f0_OK  && f1_OK)
			all_focals.push_back( sqrt( f0*f1 ) );

		//add by say
		focalsFromHomography(match_info[i].HRev, f0, f1, f0_OK, f1_OK);
		if ( f0_OK  && f1_OK)
			all_focals.push_back( sqrt( f0*f1 ) );

	}
	if (static_cast<int>(all_focals.size()) >= num_videos-1)
	{
		double median;
		std::sort(all_focals.begin(),all_focals.end());
		if (all_focals.size()%2 == 1)
			median = all_focals[all_focals.size() / 2];
		else
			median = ( all_focals[all_focals.size() / 2 - 1] + all_focals[all_focals.size() / 2] ) *0.5;

		for(int i = 0; i<num_videos; i++)
			focals[i] = median;
	}
	else
	{
		cout<<"Failed to estimate properly, use a simple way to roughly similarize"<<endl;
		double focals_sum = 0;
		for(int i=0; i<num_videos; i++)
			focals_sum += imgSet[i].cols+imgSet[i].rows;
		for(int i=0; i<num_videos; i++)
			focals[i] = focals_sum / num_videos;
	}

}

void KREstimator::calcRotation(vector<exmLog>&match_info, homoT *root, vector<CameraParams> &cameras)
{
	homoT *nowPoint;
	nowPoint = root->son;
	int picInx_pre = root->picInx;

	while(nowPoint != NULL){ //令人头痛的转换次序，应该是这样的：to-base,from-trans
		Mat_<double> K_from = Mat :: eye(3, 3, CV_64F);
		K_from(0,0) = cameras[picInx_pre].focal;
		K_from(1,1) = cameras[picInx_pre].focal * cameras[picInx_pre].aspect;
		K_from(0,2) = cameras[picInx_pre].ppx;
		K_from(1,2) = cameras[picInx_pre].ppy;

		Mat_<double> K_to = Mat :: eye(3, 3, CV_64F);
		K_to(0, 0) = cameras[nowPoint->picInx].focal;
		K_to(1, 1) = cameras[nowPoint->picInx].focal * cameras[nowPoint->picInx].aspect;
		K_to(0, 2) = cameras[nowPoint->picInx].ppx;
		K_to(1, 2) = cameras[nowPoint->picInx].ppy;

		//cout <<K_to << K_from <<endl;
		Mat R = K_to.inv() * match_info[nowPoint->matId].HRev.inv() *K_from;//change by say 2014.10.13 origin : Mat R = K_from.inv() * match_info[nowPoint->matId].HRev.inv() *K_to;

		//Mat R0 = cameras[nowPoint->picInx].R;//changed by say 2014.4.8
		cameras[nowPoint->picInx].R = cameras[picInx_pre].R * R;
		 
		//父节点的R值其实可以不用更新，否则，这迭代的初始值就有点问题
		//Mat R_ = K_to.inv() * match_info[nowPoint->matId].H.inv() * K_from; 
		//cameras[picInx_pre].R =  R0 * R_;   //changed by say 2014.4.8

		/*
		//此处添加是为了利用反方向的匹配关系计算父节点的R矩阵，2014.2.26 by LSH
		if (nowPoint->matId >= match_info.size()*0.5f)
		{
		int id = nowPoint->matId - match_info.size()/2;
		Mat R_ = K_to.inv() * match_info[id].H.inv() * K_from;
		cameras[picInx_pre].R =  R0 * R_;
		}
		else
		{
		int id = nowPoint->matId + match_info.size()/2;
		Mat R_ = K_to.inv() * match_info[id].H.inv() * K_from;
		cameras[picInx_pre].R =  R0 * R_;
		}
		*/

		//找这一层的其他节点作为现节点
		nowPoint = nowPoint->next; 
	}
	nowPoint = root->son;//这一层没有结点时转向下一层
	while(nowPoint != NULL){
		calcRotation(match_info,nowPoint,cameras);
		nowPoint = nowPoint->next;
	}
} 

bool KREstimator::estimate( vector<Mat> &imgSet, vector<imgFeatures> &imgF, vector<mLog> &matchIdx,
						  vector<CameraParams> &cameras)
{
	if(matchIdx.size()<1)   //这里小于1
	{
		cout<<"[Warrning]Links are too few, there are less than two images connected!"<<endl;
		throw sysException("[Failed]There are less than two images connected!");

/*		//cvWaitKey(0);
		//这里要只输出第一幅图像
		CameraParams camera;

		//设置为初始的相机参数

		camera.t.create(3,1,CV_32F);
		camera.t.setTo(0);
		camera.R.create(3,3,CV_32F);
		setIdentity(camera.R);

		camera.K().create(3,3,CV_32F);
		setIdentity(camera.K());

		Mat &bImg=imgSet[0];

		camera.ppx=bImg.cols/2;
		camera.ppy=bImg.rows/2;

		camera.focal=(camera.ppx+camera.ppy)*3.0;
		camera.K().at<float>(0,2)=(float)camera.ppx;
		camera.K().at<float>(1,2)=(float)camera.ppy;
		camera.K().at<float>(0,0)=(float)(camera.ppx+camera.ppy);
		camera.K().at<float>(1,1)=(float)(camera.ppy+camera.ppx);

		camera.t.at<float>(0,0)=(float)camera.ppx;
		camera.t.at<float>(0,1)=(float)camera.ppy;

		cameras.resize(1,camera);*/

		return false;
	}
	homoT* homoTRoot=new homoT;
	homoTRoot->son=NULL; //防止误访问
	homoTRoot->next=NULL;

	matcher.expandMatchInfo(imgSet,imgF,matchIdx,homoTRoot);
	vector<exmLog> &match_info = matcher.match_info;
	unsigned int video_num = (unsigned int)imgSet.size(); 
	cameras.resize(video_num);

	//粗估计
	estimatefromHomography(imgSet,match_info,homoTRoot, cameras);

	for (size_t i = 0; i < cameras.size(); ++i)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
	}

    //ray校正，计算全局RMS
	estimateBunderadjuster( imgSet, imgF, match_info, homoTRoot, cameras);

	return true;

}


void KREstimator::estimatefromHomography(vector<Mat> &imgSet,vector<exmLog> &match_info, homoT *root,
									   vector<CameraParams> &cameras)
{
	unsigned int num_videos = (unsigned int)imgSet.size();
	vector<double> focals;
	estimateFocal(imgSet,match_info,focals);

	cameras.assign(num_videos, CameraParams());

	for(unsigned int i =0; i<num_videos;++i)
		cameras[i].focal = focals[i];


	//直接调用匹配最大生成树根节点来估计rotation

	calcRotation(match_info, root, cameras);//计算旋转矩阵，需要以横向优先搜索算法来遍历

	//因为运算基于的假设是p.p在图像的中心
	for(unsigned int i=0; i<num_videos; ++i)
	{
		cameras[i].ppx += 0.5 * imgSet[i].cols;//width
		cameras[i].ppy += 0.5 * imgSet[i].rows;//height
	}

}


void KREstimator::estimateBunderadjuster(vector<Mat> &imgSet, vector<imgFeatures> &imgF,vector<exmLog> &match_info, 								    
									   homoT *root, vector<CameraParams> &cameras)
{
	num_images_ = static_cast<int>(imgSet.size());

	setUpInitialCameraParams(cameras);


	// Compute number of correspondences

	total_num_matches_ = 0;
	for (size_t i = 0; i < match_info.size(); ++i)
	{
		//changed by say
		//if( true == match_info[i].used)
		//total_num_matches_ += static_cast<int>(match_info[i].matchPointIndex.size());
		total_num_matches_ += match_info[i].num_inliers;
	}
	CvLevMarq solver(num_images_ * num_params_per_cam_,
		total_num_matches_ * num_errs_per_measurement_,
		term_criteria_);

	Mat err, jac;
	CvMat matParams = cam_params_;
	cvCopy(&matParams, solver.param);

	int iter = 0;
	for(;;)
	{
		const CvMat* _param = 0;
		CvMat* _jac = 0;
		CvMat* _err = 0;
		bool proceed = solver.update(_param, _jac, _err);
		cvCopy(_param, &matParams);

		if (!proceed || !_err)
			break;

		__debug(cout << "->";)

			if (_jac)
			{
				calcJacobian(imgSet,imgF,match_info,jac);

				__debug(cout<<"EJ;";)

					CvMat tmp = jac;
				cvCopy(&tmp, _jac);
			}

			if (_err)
			{

				calcError(imgSet,imgF,match_info,err);
				iter++;
				//if (iter>1000) throw sysException("[Failed]Iterter over max time!"); //迭代次数过大！
				CvMat tmp = err;

				double tp=sqrt(err.dot(err) / total_num_matches_);
				__debug(cout<<"E: " << tp<<" ";)
				if ((tp!=tp)          //NaN
					||(tp==-numeric_limits<double>::infinity()) //-INF
					||(tp==numeric_limits<double>::infinity())) //+INF
					throw sysException("[Failed]Buddle adjust failed!");  //
				
				cvCopy(&tmp, _err);
			}
	}

	__debug(cout<<endl;)

		__debug(
	cout<<"[Info]Bundle adjustment, final RMS error: " << sqrt(err.dot(err) / total_num_matches_)<<endl;
	cout<<"[Info]Bundle adjustment, iterations done: " << iter <<endl ;
	)

		obtainRefinedCameraParams(cameras);

	// Normalize motion to center image
		
	//这里生成最大生成树，改为直接调用
	/*	
	Graph span_tree;
	vector<int> span_tree_centers;
	findMaxSpanningTree(num_images_, pairwise_matches, span_tree, span_tree_centers);
	*/

	Mat R_inv = cameras[root->picInx].R.inv();
	for (int i = 0; i < num_images_; ++i)
		cameras[i].R = R_inv * cameras[i].R;

}


void KREstimator::setUpInitialCameraParams(const vector<CameraParams> &cameras)
{
	cam_params_.create(num_images_ * 4, 1, CV_64F);
	SVD svd;
	for (int i = 0; i < num_images_; ++i)
	{
		cam_params_.at<double>(i * 4, 0) = cameras[i].focal;

		svd(cameras[i].R, SVD::FULL_UV);
		Mat R = svd.u * svd.vt;
		if (determinant(R) < 0)
			R *= -1;

		Mat rvec;
		Rodrigues(R, rvec);
		CV_Assert(rvec.type() == CV_32F);
		cam_params_.at<double>(i * 4 + 1, 0) = rvec.at<float>(0, 0);
		cam_params_.at<double>(i * 4 + 2, 0) = rvec.at<float>(1, 0);
		cam_params_.at<double>(i * 4 + 3, 0) = rvec.at<float>(2, 0);
	}
}


void KREstimator::obtainRefinedCameraParams(vector<CameraParams> &cameras) const
{
	for (int i = 0; i < num_images_; ++i)
	{
		cameras[i].focal = cam_params_.at<double>(i * 4, 0);

		Mat rvec(3, 1, CV_64F);
		rvec.at<double>(0, 0) = cam_params_.at<double>(i * 4 + 1, 0);
		rvec.at<double>(1, 0) = cam_params_.at<double>(i * 4 + 2, 0);
		rvec.at<double>(2, 0) = cam_params_.at<double>(i * 4 + 3, 0);
		Rodrigues(rvec, cameras[i].R);

		Mat tmp;
		cameras[i].R.convertTo(tmp, CV_32F);
		cameras[i].R = tmp;
	}
}

/*   图变成树  */
void KREstimator::calcError( vector<Mat>&imgSet, vector<imgFeatures> &imgF, vector<exmLog> &match_info, 
						   Mat &err)
{

	err.create(total_num_matches_ * 3, 1, CV_64F);  //changed by say

	int new_match_idx = 0;
	int src_num =imgF.size();
	
	int *imgF_index=new int[imgSet.size()];//索引重建
	int inx = 0;
	for (int i=0;i<src_num;i++)
	{
		if (true==matcher.videosFlag[i])
		{
			imgF_index[inx] = i;
			inx++;
		}
	}


	for (unsigned int idx=0;idx<match_info.size();idx++)
	{
		//int idx = nowpoint->matId ：index of match in the matches list
			exmLog &match=match_info[idx];
			int i = match.queryInx;//nowpoint->picInx: src
			int j = match.trainInx;//root->picInx;   : dst

			double f1 = cam_params_.at<double>(i * 4, 0);
			double f2 = cam_params_.at<double>(j * 4, 0);

			double R1[9];
			Mat R1_(3, 3, CV_64F, R1);
			Mat rvec(3, 1, CV_64F);
			rvec.at<double>(0, 0) = cam_params_.at<double>(i * 4 + 1, 0);
			rvec.at<double>(1, 0) = cam_params_.at<double>(i * 4 + 2, 0);
			rvec.at<double>(2, 0) = cam_params_.at<double>(i * 4 + 3, 0);
			Rodrigues(rvec, R1_);

			double R2[9];
			Mat R2_(3, 3, CV_64F, R2);
			rvec.at<double>(0, 0) = cam_params_.at<double>(j * 4 + 1, 0);
			rvec.at<double>(1, 0) = cam_params_.at<double>(j * 4 + 2, 0);
			rvec.at<double>(2, 0) = cam_params_.at<double>(j * 4 + 3, 0);
			Rodrigues(rvec, R2_);
			
			int F_i = imgF_index[i];
			int F_j = imgF_index[j];//由于特征没有剪枝，需要回到以前的索引

			const imgFeatures& keyT = imgF[F_i];
			const imgFeatures& keyB = imgF[F_j];

		/*vector<KeyPoint> keyT,keyB;
		
		keyT = videos[match.queryInx].backGroundPoint;//query——src
		keyB = videos[match.trainInx].backGroundPoint;//train——dst*/

			const exmLog& match_info_ij = match;

			Mat_<double> K1 = Mat::eye(3, 3, CV_64F);
			K1(0,0) = f1; K1(0,2) = imgSet[i].cols * 0.5; //width
			K1(1,1) = f1; K1(1,2) = imgSet[i].rows * 0.5; //height

			Mat_<double> K2 = Mat::eye(3, 3, CV_64F);
			K2(0,0) = f2; K2(0,2) = imgSet[j].cols * 0.5;
			K2(1,1) = f2; K2(1,2) = imgSet[j].rows * 0.5;

			Mat_<double> H1 = R1_ * K1.inv();
			Mat_<double> H2 = R2_ * K2.inv();

			for (size_t k = 0; k < match_info_ij.matchPointIndex.size(); ++k)
			{
				if (!match_info_ij.inliers_mask[k])//add by LSH. 2014.3.4
					continue;

				const DMatch &m =  match_info_ij.matchPointIndex[k];

			//Point2f p1 = features1.keypoints[m.queryIdx].pt;
				Point2f p1 = keyT.backGroundPoint[m.queryIdx].pt;
				double x1 = H1(0,0)*p1.x + H1(0,1)*p1.y + H1(0,2);
				double y1 = H1(1,0)*p1.x + H1(1,1)*p1.y + H1(1,2);
				double z1 = H1(2,0)*p1.x + H1(2,1)*p1.y + H1(2,2);
				double len = sqrt(x1*x1 + y1*y1 + z1*z1);
				x1 /= len; y1 /= len; z1 /= len;

			//Point2f p2 = features2.keypoints[m.trainIdx].pt;
				Point2f p2 = keyB.backGroundPoint[m.trainIdx].pt;
				double x2 = H2(0,0)*p2.x + H2(0,1)*p2.y + H2(0,2);
				double y2 = H2(1,0)*p2.x + H2(1,1)*p2.y + H2(1,2);
				double z2 = H2(2,0)*p2.x + H2(2,1)*p2.y + H2(2,2);
				len = sqrt(x2*x2 + y2*y2 + z2*z2);
				x2 /= len; y2 /= len; z2 /= len;

				double mult = sqrt(f1 * f2);
				err.at<double>(3 * new_match_idx, 0) = mult * (x1 - x2);
				err.at<double>(3 * new_match_idx + 1, 0) = mult * (y1 - y2);
				err.at<double>(3 * new_match_idx + 2, 0) = mult * (z1 - z2);

				new_match_idx++;
			}
		
	}
	delete []imgF_index;
}


void KREstimator::calcJacobian( vector<Mat>&imgSet, vector<imgFeatures> &imgF, vector<exmLog> &match_info, 
							  Mat &jac)
{
	jac.create(total_num_matches_ * 3, num_images_ * 4, CV_64F);
	double val;
	const double step = 1e-3;

	for (int i = 0; i < num_images_; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			val = cam_params_.at<double>(i * 4 + j, 0);
			cam_params_.at<double>(i * 4 + j, 0) = val - step;
			calcError(imgSet,imgF,match_info,err1_);
			cam_params_.at<double>(i * 4 + j, 0) = val + step;
			calcError(imgSet,imgF,match_info,err2_);
			calcDeriv(err1_, err2_, 2 * step, jac.col(i * 4 + j));
			cam_params_.at<double>(i * 4 + j, 0) = val;
		}
	}
}
