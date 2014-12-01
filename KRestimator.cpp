#include "KRestimator.h"

/********************************* match points further process***************************************/

int OptimizeMatch::expandMatchInfo(vector<Mat> &imgSet, vector<imgFeatures> &imgF,vector<mLog> &matchIdx, homoT *root)
{
	unsigned int videoNums = (unsigned int)imgSet.size();
	videosFlag.resize(videoNums);
	match_info.resize( matchIdx.size());
	calAllMatch(imgSet,imgF,matchIdx);
	
	// ����ƥ����
		
	__debug(
			cout << "[Info]Generate match tree from match set..." << endl;
		)
	
	//�������ƥ���ϵ����������ƥ���ϵ��query��train
	MaxSpanningTree(root,videoNums); 

	adjustVideos(imgSet);

	//��������ֵ
 	return adjustIndx( root, videoNums);

}




//����任�����ƥ���ϵ,��Ҫ��queryת����trainͼ���ת������H����
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

		keyT = imgF[matchIdx[id].queryInx].backGroundPoint;//query����src
		keyB = imgF[matchIdx[id].trainInx].backGroundPoint;//train����dst

		float width_qI  = imgSet[matchIdx[id].queryInx].cols*0.5f;//�������ᵽ�¸�forѭ����ǰ�������Լ���������
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

		match_info[id].H = findHomography(src,dst,match_info[id].inliers_mask,CV_RANSAC,3.0); //Ĭ����Ϊ3.0
		match_info[id].num_inliers = 0;
		for (size_t z = 0; z < match_info[id].inliers_mask.size(); ++z)
		{
			if (match_info[id].inliers_mask[z])
				match_info[id].num_inliers++;
		}

		//���㷴�任
		match_info[id].HRev= findHomography(dst,src,match_info[id].inliers_mask_rev,CV_RANSAC,3.0); //Ĭ����Ϊ3.0
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
	//ɸѡ���Ŷȳ���һ����ֵ��ƥ���ϵ
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


    //����������㷨������Kruskal�㷨
    sort(match_info.begin(),match_info.end());  //����������operator �� ��������,ȨֵΪƥ������Ŀ
    int *setCount=new int[video_num];
    for (int k=0;k<video_num;k++) 
		setCount[k]=0;


    int setFlag=1;

    for (int k=(match_info.size()-1);k>=0;k--)
    {
        if ((setCount[match_info[k].queryInx]==0)&&(setCount[match_info[k].trainInx]==0))
		{   //�����㶼û�м������ɼ��ϣ��������������һ���¼���
            setCount[match_info[k].queryInx]=setFlag;
            setCount[match_info[k].trainInx]=setFlag;
            setFlag++;
            match_info[k].inTree=true;  //�����߼��뼯��
			match_info[k].used = false;
        }else if ((setCount[match_info[k].queryInx]==0)&&(setCount[match_info[k].trainInx]!=0))
		{ //����һ����û���뼯�ϣ������
            setCount[match_info[k].queryInx]=setCount[match_info[k].trainInx];
            match_info[k].inTree=true;  //�����߼��뼯��
			match_info[k].used = false;
        }
		else if ((setCount[match_info[k].queryInx]!=0)&&(setCount[match_info[k].trainInx]==0))
		{ //����һ����û���뼯�ϣ������
            setCount[match_info[k].trainInx]=setCount[match_info[k].queryInx];
            match_info[k].inTree=true;  //�����߼��뼯��
			match_info[k].used = false;
        }
		else if (setCount[match_info[k].queryInx]!=setCount[match_info[k].trainInx])
		{   //����������˲�ͬ���ϣ����������������ϣ������ϴ���ͳһ��
            match_info[k].inTree=true;  //�����߼��뼯��
			match_info[k].used=false;
            int needChange=setCount[match_info[k].trainInx];
            int chage=setCount[match_info[k].queryInx];
            for (int w=0;w<video_num;w++){
                if (setCount[w]==needChange){
                    setCount[w]=chage;
                }
            } //���¼��ϵĴ���
        } //������������������
		match_info[k].used=false;
    }
    delete [] setCount;

/*	// Ѱ��ƥ�����Ŀ����ͼƬ
	//�����һ���뷨���������Ѱ��ʹ������ĵ��ǲ��ǿ��Ը��ã������Ǹ���~���ԸϽ���
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
	//������ʵ������㷨��
	//����ͨ�����޻�ͼת���������С�����Ĵ��룺
	//ÿ��ɾ�����е�Ҷ�ڵ㣬ֱ��ֻʣ��һ�����������ڵ�Ϊֹ

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

	// �����ͼƬ��ƽ����Ϊ��ƽ��
	__debug(
		cout << "[Info]Setting video input "<< baseImgFact <<" as base img..." << endl;
	)

	homoTRoot->picInx=baseImgFact;
	homoTRoot->next=NULL;
	videosFlag[baseImgFact]=true;

	delete [] imgInd;

	// ����ƥ����
	__debug(
		cout << "[Info]Building match tree from match set..." << endl;
	)
		buildHomoTree(homoTRoot); //�������ƥ���ϵ����������ƥ���ϵ��query��train
    __debug(printTree(homoTRoot);)

	__debug(
	cout << "[Info]Adjusting match index and videos index..." << endl;
	)

}


//���ҵ��Ļ�׼ƽ������������ƥ����2014.2.20
//ͬʱ��������ƥ����Ϣ������matchIdx
//ʹ�����е�nowPoint->picInx =  matchIdx[nowPoint->matId].query
int OptimizeMatch::buildHomoTree( homoT *root)
{
	int rootId=root->picInx;
	int sonNum=0;
	homoT *nowPoint= NULL;
	//��ʼ����
	root->son=nowPoint;
	unsigned int matchSize=(unsigned int)match_info.size();

	for (unsigned int i=0;i<matchSize;i++)
	{
		if ((match_info[i].inTree==true)&&(match_info[i].used==false))   //����ֻ�Դ��������ı߽��в���
		{
			if ((match_info[i].queryInx == rootId)||(match_info[i].trainInx == rootId))
			{
				if (match_info[i].queryInx == rootId)
				{   //���queryInx�Ǹ��ڵ㣬��ô�������matchIdx�Ķ�����
					exmLog matTemp;
					matTemp.trainInx = match_info[i].queryInx;
					matTemp.queryInx = match_info[i].trainInx;
					matTemp.matchPointIndex = match_info[i].matchPointIndexRev;
					matTemp.matchPointIndexRev =match_info[i].matchPointIndex;
					matTemp.H =match_info[i].HRev;  //��߲���clone������ָ����Զ�ѡ���Ƿ��ͷſռ�
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


//videoAdjust�ᱻ����������С
int OptimizeMatch::adjustVideos(vector<Mat> &videoAdjust)
{
	vector<Mat> videosNew;

	for (unsigned int i=0;i<videoAdjust.size();i++)
	{
		if (videosFlag[i]==true)
		{  //��������Ƶ������
			videosNew.push_back(videoAdjust[i]);
		}
	}

	videoAdjust=videosNew;
	return 1;
}

//��������ֵ����videoSeq��ɾȥû�������е�videos������ֵΪ����֮���videos�ĸ�����
int OptimizeMatch::adjustIndx(homoT *homoTRoot, unsigned int videoSize)
{
	int *indxMap=new int[videoSize];

	int inx=0;
	for (unsigned int i=0;i<videoSize;i++)
	{
		if (videosFlag[i]==true)
		{  //��������Ƶ������
			indxMap[i]=inx;
			inx++;
		}else{
			indxMap[i]=-1;
		}
	}

	int ans=inx-1;

	for (unsigned int i=0;i<match_info.size();i++){    //����mlog���������
		match_info[i].queryInx=indxMap[match_info[i].queryInx];
		match_info[i].trainInx=indxMap[match_info[i].trainInx];
	}

	int *mlogInxMap=new int[match_info.size()];  //��mlog����ɾ��,ɾ����Щ����û���κ�һ���ڵ������ӵ�ƥ���ϵ

	vector<exmLog> newMathcIdx;
	inx=0;
	for (unsigned int i=0;i<match_info.size();i++){
		if ((match_info[i].queryInx>=0)&&(match_info[i].trainInx>=0)){  //���û������
			newMathcIdx.push_back(match_info[i]);
			mlogInxMap[i]=inx;
			inx++;
		}else{
			mlogInxMap[i]=-1;
		}
	}

	match_info=newMathcIdx;
	newMathcIdx.clear();

	//������idx��matId���и���

	homoTRoot->picInx=indxMap[homoTRoot->picInx];

	updateTreIdx(homoTRoot,indxMap,mlogInxMap);

	delete [] indxMap;   //�ͷſռ�
	delete [] mlogInxMap;

	return ans;
}

int OptimizeMatch::updateTreIdx(homoT *root, int idxMap[], int matIdxMap[])
{
	homoT *nowPoint;

	nowPoint=root->son;
	while (nowPoint!=NULL)
	{
		nowPoint->matId=matIdxMap[nowPoint->matId];  //����ӳ���ϵ��������
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

	while(nowPoint != NULL){ //����ͷʹ��ת������Ӧ���������ģ�to-base,from-trans
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
		 
		//���ڵ��Rֵ��ʵ���Բ��ø��£�����������ĳ�ʼֵ���е�����
		//Mat R_ = K_to.inv() * match_info[nowPoint->matId].H.inv() * K_from; 
		//cameras[picInx_pre].R =  R0 * R_;   //changed by say 2014.4.8

		/*
		//�˴������Ϊ�����÷������ƥ���ϵ���㸸�ڵ��R����2014.2.26 by LSH
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

		//����һ��������ڵ���Ϊ�ֽڵ�
		nowPoint = nowPoint->next; 
	}
	nowPoint = root->son;//��һ��û�н��ʱת����һ��
	while(nowPoint != NULL){
		calcRotation(match_info,nowPoint,cameras);
		nowPoint = nowPoint->next;
	}
} 

bool KREstimator::estimate( vector<Mat> &imgSet, vector<imgFeatures> &imgF, vector<mLog> &matchIdx,
						  vector<CameraParams> &cameras)
{
	if(matchIdx.size()<1)   //����С��1
	{
		cout<<"[Warrning]Links are too few, there are less than two images connected!"<<endl;
		throw sysException("[Failed]There are less than two images connected!");

/*		//cvWaitKey(0);
		//����Ҫֻ�����һ��ͼ��
		CameraParams camera;

		//����Ϊ��ʼ���������

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
	homoTRoot->son=NULL; //��ֹ�����
	homoTRoot->next=NULL;

	matcher.expandMatchInfo(imgSet,imgF,matchIdx,homoTRoot);
	vector<exmLog> &match_info = matcher.match_info;
	unsigned int video_num = (unsigned int)imgSet.size(); 
	cameras.resize(video_num);

	//�ֹ���
	estimatefromHomography(imgSet,match_info,homoTRoot, cameras);

	for (size_t i = 0; i < cameras.size(); ++i)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
	}

    //rayУ��������ȫ��RMS
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


	//ֱ�ӵ���ƥ��������������ڵ�������rotation

	calcRotation(match_info, root, cameras);//������ת������Ҫ�Ժ������������㷨������

	//��Ϊ������ڵļ�����p.p��ͼ�������
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
				//if (iter>1000) throw sysException("[Failed]Iterter over max time!"); //������������
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
		
	//���������������������Ϊֱ�ӵ���
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

/*   ͼ�����  */
void KREstimator::calcError( vector<Mat>&imgSet, vector<imgFeatures> &imgF, vector<exmLog> &match_info, 
						   Mat &err)
{

	err.create(total_num_matches_ * 3, 1, CV_64F);  //changed by say

	int new_match_idx = 0;
	int src_num =imgF.size();
	
	int *imgF_index=new int[imgSet.size()];//�����ؽ�
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
		//int idx = nowpoint->matId ��index of match in the matches list
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
			int F_j = imgF_index[j];//��������û�м�֦����Ҫ�ص���ǰ������

			const imgFeatures& keyT = imgF[F_i];
			const imgFeatures& keyB = imgF[F_j];

		/*vector<KeyPoint> keyT,keyB;
		
		keyT = videos[match.queryInx].backGroundPoint;//query����src
		keyB = videos[match.trainInx].backGroundPoint;//train����dst*/

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
