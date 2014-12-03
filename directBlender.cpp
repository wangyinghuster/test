#include "directBlender.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp> 

//������floodPointΪ���ӵ㣬floodPoint����ֵΪ��������ͨ��
//��ͨ��ĵ㼯������neiPoints��
//��ͨ�������󣬽�ͼ���е���ͨ��������Ϊlabel
//ע�⣺labelҪ��floodPoint����ֵ��ͬ
static void NeighbourSearch(Mat& src, Point2i floodPoint, vector<Point2i>& neiPoints, int label)
{
	int value = src.at<unsigned char>(floodPoint.y, floodPoint.x);
	int bottom = 0,top=1;
	Point2i tmp;
	neiPoints.clear();
	neiPoints.push_back(floodPoint);
	src.at<unsigned char>(floodPoint.y, floodPoint.x) = label;
	while(bottom!=top){
		tmp = neiPoints[bottom++];
		if(tmp.y>0&&src.at<unsigned char>(tmp.y-1, tmp.x)==value){
			neiPoints.push_back(Point2i(tmp.x,tmp.y-1));
			src.at<unsigned char>(tmp.y-1, tmp.x) =label;//���ʹ�
			top++;
		}
		if(tmp.y<src.rows-1 && src.at<unsigned char>(tmp.y+1, tmp.x)==value){
			neiPoints.push_back(Point2i(tmp.x,tmp.y+1));
			src.at<unsigned char>(tmp.y+1, tmp.x) = label;//���ʹ�
			top++;
		}
		if(tmp.x>0&&src.at<unsigned char>(tmp.y, tmp.x-1)==value){
			neiPoints.push_back(Point2i(tmp.x-1,tmp.y));
			src.at<unsigned char>(tmp.y, tmp.x-1) = label;//���ʹ�
			top++;
		}
		if(tmp.x<src.cols-1 && src.at<unsigned char>(tmp.y, tmp.x+1)==value){
			neiPoints.push_back(Point2i(tmp.x+1,tmp.y));
			src.at<unsigned char>(tmp.y, tmp.x+1) = label;//���ʹ�
			top++;
		}
		if(tmp.x<src.cols-1 && tmp.y<src.rows-1 && src.at<unsigned char>(tmp.y+1, tmp.x+1)==value){
			neiPoints.push_back(Point2i(tmp.x+1,tmp.y+1));
			src.at<unsigned char>(tmp.y+1, tmp.x+1) = label;//���ʹ�
			top++;
		}
		if(tmp.x<src.cols-1 && tmp.y<src.rows-1 && src.at<unsigned char>(tmp.y+1, tmp.x+1)==value){
			neiPoints.push_back(Point2i(tmp.x+1,tmp.y+1));
			src.at<unsigned char>(tmp.y+1, tmp.x+1) = label;//���ʹ�
			top++;
		}
		if(tmp.x<src.cols-1 && tmp.y<src.rows-1 && src.at<unsigned char>(tmp.y+1, tmp.x+1)==value){
			neiPoints.push_back(Point2i(tmp.x+1,tmp.y+1));
			src.at<unsigned char>(tmp.y+1, tmp.x+1) = label;//���ʹ�
			top++;
		}
		if(tmp.x<src.cols-1 && tmp.y>0 && src.at<unsigned char>(tmp.y-1, tmp.x+1)==value){
			neiPoints.push_back(Point2i(tmp.x+1,tmp.y-1));
			src.at<unsigned char>(tmp.y-1, tmp.x+1) = label;//���ʹ�
			top++;
		}
		if(tmp.x>0 && tmp.y<src.rows-1 && src.at<unsigned char>(tmp.y+1, tmp.x-1)==value){
			neiPoints.push_back(Point2i(tmp.x-1,tmp.y+1));
			src.at<unsigned char>(tmp.y+1, tmp.x-1) = label;//���ʹ�
			top++;
		}
		if(tmp.x>0 && tmp.y>0 && src.at<unsigned char>(tmp.y-1, tmp.x-1)==value){
			neiPoints.push_back(Point2i(tmp.x-1,tmp.y-1));
			src.at<unsigned char>(tmp.y-1, tmp.x-1) = label;//���ʹ�
			top++;
		}
	}
}


void directBlender::prepare(vector<Mat> &warpSeamMask, vector<Mat> &warpMask, vector<Point> &topleft)
{
	//����Ҫ��Mask����CV_8UC1������
    int imgNum=warpMask.size();
    //��������tl�������յ�ͼ���С,����Ҫ��tl��Ϊ��ֵ
    getAllSize(warpMask,topleft);
	imgMaskWeight.clear();
	seamMasks_.clear();
	grayImgs_.clear();
    for (int i=0;i<imgNum;i++){
		Mat T = Mat::zeros(warpMask[i].size(), CV_8U);
		foreMasks_.push_back(T);
		warpForeMasks_.push_back(T);
		for (int y = 0; y < T.rows; ++y){
			for (int x = 0; x < T.cols; ++x){				
				if(warpMask[i].at<unsigned char>(y, x) > 0)
					T.at<unsigned char>(y, x) = 1;
			}
		}
		imgMaskWeight.push_back(T);
	}
	for (int i=0;i<imgNum;i++){
		seamMasks_.push_back(warpSeamMask[i].clone());//����warpSeamMask��adjustForgroundʹ��
	}
//	foregroudMap=Mat::zeros(outRows,outCols,CV_8UC1);//��ʼ��ȫ��ǰ��map
    prepared=true;
}

void directBlender::doBlend(vector<Mat> &warpImg,vector<Point> &topleft,Mat &panoImgOut)
{
	int imgNum=warpImg.size();
	if (!prepared) {
		throw sysException("Need to initialize before blend!");
	}
	if(grayImgs_.size()==0){
		foreOut_ = Mat::zeros(outRows,outCols,CV_8UC1);
		grayImgOut_ = Mat::zeros(outRows,outCols,CV_8UC1);
		grayImgs_.clear();
		for(int i=0;i<imgNum;i++){
			Mat &weightMap=imgMaskWeight[i];
			Mat &imgNow = warpImg[i];
			Mat grayImg;
			cvtColor(warpImg[i], grayImg,CV_BGR2GRAY);
			grayImgs_.push_back(grayImg);
			Point &topLeft=topleft[i];
			for (int r=0;r<imgNow.rows;r++){
				unsigned char* maskWeight = weightMap.ptr<unsigned char>(r);
				unsigned char* grayNowRow = grayImgs_[i].ptr<unsigned char>(r);
				unsigned char* grayOut = grayImgOut_.ptr<unsigned char>(r+topLeft.y);
				int xOF=topLeft.x;
				int sum;
				float w1,w2;
				for (int c=0;c<imgNow.cols;c++){
						if(maskWeight[c]>0){
						if (grayOut[c+xOF]==0){
							grayOut[c+xOF] = grayNowRow[c];
						}else{
							sum = grayOut[c+xOF]+grayNowRow[c];
							w1 = (float)grayOut[c+xOF]/sum;
							w2 = (float)grayNowRow[c]/sum;
							grayOut[c+xOF] = (unsigned char)(grayOut[c+xOF]*w1+grayNowRow[c]*w2);
						}
					}
				}
			}
		}	
	}

	Mat grayImgOut = Mat::zeros(outRows,outCols,CV_8UC1);
	panoImgOut=Mat::zeros(outRows,outCols,CV_8UC3);  //RGB��ʽ,����ͼ��,���Ҫ��ʼ��Ϊ�㰡�������ֵܣ���������
	for (int i=0;i<imgNum;i++){
		Point &topLeft=topleft[i];
		Mat &imgNow = warpImg[i];
		Mat grayNow;
		cvtColor(warpImg[i], grayNow, CV_BGR2GRAY);
		for (int r=0;r<imgNow.rows;r++){
			unsigned char* grayOutSaved = grayImgOut_.ptr<unsigned char>(r+topLeft.y);
			unsigned char* grayImgSaved = grayImgs_[i].ptr<unsigned char>(r);
			unsigned char* foreOutMask = foreOut_.ptr<unsigned char>(r+topLeft.y);//�洢�ŻҶ�ֵ���ܸ��µ��ɰ�
			unsigned char* foreImgMask = foreMasks_[i].ptr<unsigned char>(r);//�洢��ǰ���ɰ�
			unsigned char* maskWeight = imgMaskWeight[i].ptr<unsigned char>(r);
			unsigned char* warpFore = warpForeMasks_[i].ptr<unsigned char>(r);

			unsigned char* grayOutNow = grayImgOut.ptr<unsigned char>(r+topLeft.y);
			unsigned char* grayImgNow = grayNow.ptr<unsigned char>(r);
			Point3_<unsigned char>* imgNowRow = imgNow.ptr<Point3_<unsigned char> >(r);
			Point3_<unsigned char>* outRow = panoImgOut.ptr<Point3_<unsigned char> >(r+topLeft.y);
			int xOF=topLeft.x; 
			float w1,w2;
			for (int c=0;c<imgNow.cols;c++){
				if(maskWeight[c]>0){
					if(foreOutMask[c+xOF]){//���ܸ���
						if(foreImgMask[c]==0)//0���� 1Ĩ��
						{
							float div = 1;
							if(grayImgSaved[c]>0)
							   div = fabs((double)(grayImgNow[c]-grayImgSaved[c])/grayImgSaved[c]);
							if(div>1)
								w1 = 1;
							else{
								w1 = (grayImgSaved[c]==0)?1:((float)grayOutSaved[c+xOF]/grayImgSaved[c]);
								unsigned char max = (imgNowRow[c].x>imgNowRow[c].y)?imgNowRow[c].x:imgNowRow[c].y;
								max = (max>imgNowRow[c].z)?max:imgNowRow[c].z;
								if(max==0)
									w1 = 1;
								else if(w1>255.0/max)
									w1 = 1;
							}
							outRow[c+xOF].x =(unsigned char)imgNowRow[c].x*w1;
							outRow[c+xOF].y =(unsigned char)imgNowRow[c].y*w1;
							outRow[c+xOF].z =(unsigned char)imgNowRow[c].z*w1;
						}
					}
					else{
						if (grayOutNow[c+xOF]==0){
						outRow[c+xOF].x =imgNowRow[c].x;
						outRow[c+xOF].y =imgNowRow[c].y;
						outRow[c+xOF].z =imgNowRow[c].z;
						grayOutNow[c+xOF] = grayImgNow[c];
						}else{
							w1 = (float)grayOutNow[c+xOF]/(grayOutNow[c+xOF]+grayImgNow[c]);
							w2 = (float)grayImgNow[c]/(grayOutNow[c+xOF]+grayImgNow[c]);

							outRow[c+xOF].x = (unsigned char)(outRow[c+xOF].x*w1 + imgNowRow[c].x*w2);
							outRow[c+xOF].y = (unsigned char)(outRow[c+xOF].y*w1 + imgNowRow[c].y*w2);
							outRow[c+xOF].z = (unsigned char)(outRow[c+xOF].z*w1 + imgNowRow[c].z*w2);
							grayOutNow[c+xOF] = (unsigned char)(grayOutNow[c+xOF]*w1+grayImgNow[c]*w2);
						}
						//����
						grayImgSaved[c] = grayImgNow[c];
						grayOutSaved[c+xOF] = grayOutNow[c+xOF];

					}
						
				}
					
			}
		}
	}
	

	foreOut_.release();
	
}


void directBlender::adjustForground(vector<Mat> &warpImg,vector<Mat> warpForeMsk,vector<Point> topleft)
{
	int imgNum=warpForeMsk.size();
	foreMasks_.clear();//ÿ��ͼ����Ҫ���⴦���ǰ����ͨ��  =1,Ĩ�� =0����
	warpForeMasks_.clear();//����warpForeMsk��doBlendʹ��
	foreOut_ = Mat::zeros(outRows,outCols,CV_8UC1);//ȫ����Ҫ���⴦���ǰ����ͨ�� =1 ����foreMask������ =0 �����⴦��
	vector<vector<Point2i>> vecNeiPoints;//���洩���ص�����Ե����ͨ��
	vector<Point2i> neiPoints;
	vector<pair<int, int>> maskIndex;
	for (int i=0;i<imgNum;i++){
		foreMasks_.push_back(Mat::zeros(warpForeMsk[i].size(),warpForeMsk[i].type()));
		warpForeMasks_.push_back(warpForeMsk[i].clone());//����ԭʼ�ı�����ģ�õ���ǰ��mask���Ժ�����ò���Ҫɾ����������
		if (warpForeMsk[i].size()!=warpImg[i].size()) throw sysException("Different size of warpedImg and warpedImgMask");
		if (warpForeMsk[i].type()!=CV_8UC1) throw sysException("Type of foreground mask matrix is worng! Should be CV_8UC1!");

		//��ֵ��
		threshold(warpForeMsk[i],warpForeMsk[i],10.0f,255,0);
	}
	vecNeiPoints.clear();
	maskIndex.clear();
	for(int i=0;i<imgNum;i++){		
		neiPoints.clear();		
		Mat& fmsk = warpForeMsk[i];
		Mat& seamMask = seamMasks_[i];
		//Ѱ�ҿ�Խ�ص�����Ե����ͨ��
		for (int r=0;r<fmsk.rows;++r){
			unsigned char *tptr=fmsk.ptr<unsigned char>(r);  //Ҫ���ɵ���ͨ��ģ��
			unsigned char *seamMaskRow=seamMask.ptr<unsigned char>(r);
			for (int c=0;c<fmsk.cols;++c){
				if (tptr[c]==0xFF && seamMaskRow[c]>0){ //ǰ�� ����ص����߽� //125������ֵ�ƺ�û����֮����ɾ��     ע������ط��ǲ��Ե� 125Ӧ����seammask���жϣ���ֻ�ǲ���
					NeighbourSearch(fmsk,Point2i(c,r),neiPoints,vecNeiPoints.size()+1);
					//�жϸ���ͨ���Ƿ�ȫ�� ���Ͻϵ�ͼ�� ���ص����ڣ����Ƿ�ȫ��125
					int flag = -1;
					for(int ii=0;ii<neiPoints.size();ii++){
						int value = seamMask.at<unsigned char>(neiPoints[ii].y, neiPoints[ii].x);
						if(value>0&&value!=125){
							flag=255-value;
							break;
						}
					}
					if(flag==-1){//flag>0˵����ͨ���Ӧ��seamMask���в���125��ֵ������ȫ���ص�����,flag=-1��ȫ���ص����ҪĨ��
						for(int ii=0;ii<neiPoints.size();ii++){
							int y = neiPoints[ii].y;
							int x = neiPoints[ii].x;
							foreMasks_[i].at<unsigned char>(y, x) = 1;//Ĩ��
							foreOut_.at<unsigned char>(y+topleft[i].y, x+topleft[i].x)= 1;
						}
					}
					else{
						vector<Point2i> points = neiPoints;
						vecNeiPoints.push_back(points);
						if(flag>imgNum)throw sysException("flag");
						maskIndex.push_back(make_pair(i, flag));//flag��ͼ��i��Ӧ��ͼ��index,���ǹ�ͬ���������ص���
					}
				}
			}
		}
	}
	//Ѱ�����ϵ͵�ͼ���е���ͨ�� ��Ӧ�� ���ϸߵ�ͼ���е���ͨ��
	for(int k=0;k<vecNeiPoints.size();k++){
		if(maskIndex[k].first!=2){//�ж��Ƿ������ϵ͵�
			int i = maskIndex[k].first;
			int index = maskIndex[k].second;
			//�ڶ�Ӧͼ���ϵ�ӳ��
			int deltax = topleft[i].x-topleft[index].x;
			int deltay = topleft[i].y-topleft[index].y;
			vector<Point2i> mapPoints;
			Mat map = Mat::zeros(warpForeMsk[index].size(),warpForeMsk[index].type());//�洢ͼ��i��ǰ����ͼ��index��ӳ��						
			mapPoints.clear();
			int width = warpForeMsk[index].cols,height = warpForeMsk[index].rows;
			for(int ii=0;ii<vecNeiPoints[k].size();ii++){
				int y = vecNeiPoints[k][ii].y+deltay, x = vecNeiPoints[k][ii].x+deltax;
				if((y>0&&y<height)&&(x>0&&x<width)){
					mapPoints.push_back(Point2i(x,y));
					map.at<unsigned char>(y, x) = 255;
				}
			}
			if(mapPoints.size()>0){
				//����֮��Ӧ��ͼ����Ѱ�Ҿ����ص�����������ͨ�����ж���ͨ���Ƿ񴩹��ص�����Ե
				int overCnt = 0;
				int maxOverIndex = -1;
				vector<Point2i> maxOverlap;
				Mat temp = warpForeMsk[index].clone();
				for(int ii=0;ii<mapPoints.size();ii++){
					int tmpCnt=0;
					int value = temp.at<unsigned char>(mapPoints[ii].y, mapPoints[ii].x);
					if(value==0xff){
						neiPoints.clear();
						NeighbourSearch(temp, Point2i(mapPoints[ii].x,mapPoints[ii].y), neiPoints,0);
						//neiPoints��map�ص�����
						for(int jj=0;jj<neiPoints.size();jj++)
							if(map.at<unsigned char>(neiPoints[jj].y, neiPoints[jj].x))
								tmpCnt++;
						if(tmpCnt>overCnt){
							overCnt = tmpCnt;
							maxOverlap.clear();
							maxOverlap = neiPoints;
						}
					}
					else if(value>0){
						if(value>maskIndex.size())throw sysException("value");
						int loc = value-1;						
						if ((maskIndex[k].first!=maskIndex[loc].second)||(maskIndex[loc].first!=maskIndex[k].second)) throw sysException("maskIndex[k].first!=maskIndex[loc].second");
						for(int jj=0;jj<vecNeiPoints[loc].size();jj++)
							if(map.at<unsigned char>(vecNeiPoints[loc][jj].y, vecNeiPoints[loc][jj].x)){
								tmpCnt++;
								temp.at<unsigned char>(vecNeiPoints[loc][jj].y, vecNeiPoints[loc][jj].x)=0;
							}
						if(tmpCnt>overCnt){
							overCnt = tmpCnt;
							maxOverlap.clear();
							maxOverlap = vecNeiPoints[loc];
							maxOverIndex = loc;
						}
					}
				 }//end of for(int ii=0
				 if(overCnt>0){//�ڶ�Ӧͼ�����ҵ�����ͨ��
					if(maxOverIndex>0){//��Ӧͼ������ͨ�򴩹��ص����߽磬������Ӧͼ���ϵ���ͨ��
						//����ͼ���ϵ�ǰ��Ĩ��
						for(int ii=0;ii<vecNeiPoints[k].size();ii++){
							int y = vecNeiPoints[k][ii].y;
							int x = vecNeiPoints[k][ii].x;
							foreMasks_[i].at<unsigned char>(y, x) = 1;
							foreOut_.at<unsigned char>(y+topleft[i].y, x+topleft[i].x)= 1;							
						}
						//����Ӧͼ��ǰ������Ϊ���⴦�����ҵ���Ӧͼ���ڸ�ͼ���ϵ�ӳ��
						int width = warpForeMsk[i].cols,height = warpForeMsk[i].rows;
						for(int ii=0;ii<maxOverlap.size();ii++){
							int y = maxOverlap[ii].y;
							int x = maxOverlap[ii].x;
							foreOut_.at<unsigned char>(y+topleft[index].y, x+topleft[index].x)= 1;							
							 y -= deltay;
							 x -= deltax;
							if((y>=0&&y<height)&&(x>=0&&x<width)){
								foreMasks_[i].at<unsigned char>(y, x) = 1;
								foreOut_.at<unsigned char>(y+topleft[i].y, x+topleft[i].x)= 1;														 
							}				
						}

						continue;
					}
					else{////��Ӧͼ������ͨ��û�д����ص����߽磬������ͼ���ϵ���ͨ�����Ͻϵ͵ģ�
						//����Ӧͼ���ϵ���ͨ��Ĩ��
						int width = warpForeMsk[i].cols,height = warpForeMsk[i].rows;
						for(int jj=0;jj<maxOverlap.size();jj++){
							int y = maxOverlap[jj].y, x= maxOverlap[jj].x;
							foreMasks_[index].at<unsigned char>(y, x) = 1;
							foreOut_.at<unsigned char>(y+topleft[index].y,x+topleft[index].x)=1;
							y -= deltay;
							x -= deltax;
							if((y>=0&&y<height)&&(x>=0&&x<width)){
								foreMasks_[i].at<unsigned char>(y, x) = 0;//��ֹ���ֿն�,������Ӧͼ��Ĩ���Ļ���Ҫ�ָ�һ��														 
							}
						}						
					}
					
				 }
			
			}//end of if(mapPoints)
			//�ڶ�Ӧͼ����û��ӳ��� �� �ڶ�Ӧͼ����û�ҵ���ͨ����������
			for(int ii=0;ii<vecNeiPoints[k].size();ii++){
				int y = vecNeiPoints[k][ii].y;
				int x = vecNeiPoints[k][ii].x;
				//foreMasks_[i].at<unsigned char>(y, x) = 0;//�ڸ�ͼ���ϵı���
				foreOut_.at<unsigned char>(y+topleft[i].y, x+topleft[i].x)= 1;
			}//end of for(int ii = 0
			if(mapPoints.size()>0){//�ڶ�Ӧͼ���ϵ�ӳ��Ĩ��
				for(int jj=0;jj<mapPoints.size();jj++){
					int y = mapPoints[jj].y, x= mapPoints[jj].x;
					foreMasks_[index].at<unsigned char>(y, x) = 1;
					foreOut_.at<unsigned char>(y+topleft[index].y,x+topleft[index].x)=1;	
				}
			}
		}//end of if(maskIndex)			
	}//end of for(int k=0)
}