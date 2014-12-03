#include "directBlender.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp> 

//查找以floodPoint为种子点，floodPoint处的值为条件的连通域
//连通域的点集保存在neiPoints里
//连通域遍历完后，将图像中的连通域区域标记为label
//注意：label要与floodPoint处的值不同
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
			src.at<unsigned char>(tmp.y-1, tmp.x) =label;//访问过
			top++;
		}
		if(tmp.y<src.rows-1 && src.at<unsigned char>(tmp.y+1, tmp.x)==value){
			neiPoints.push_back(Point2i(tmp.x,tmp.y+1));
			src.at<unsigned char>(tmp.y+1, tmp.x) = label;//访问过
			top++;
		}
		if(tmp.x>0&&src.at<unsigned char>(tmp.y, tmp.x-1)==value){
			neiPoints.push_back(Point2i(tmp.x-1,tmp.y));
			src.at<unsigned char>(tmp.y, tmp.x-1) = label;//访问过
			top++;
		}
		if(tmp.x<src.cols-1 && src.at<unsigned char>(tmp.y, tmp.x+1)==value){
			neiPoints.push_back(Point2i(tmp.x+1,tmp.y));
			src.at<unsigned char>(tmp.y, tmp.x+1) = label;//访问过
			top++;
		}
		if(tmp.x<src.cols-1 && tmp.y<src.rows-1 && src.at<unsigned char>(tmp.y+1, tmp.x+1)==value){
			neiPoints.push_back(Point2i(tmp.x+1,tmp.y+1));
			src.at<unsigned char>(tmp.y+1, tmp.x+1) = label;//访问过
			top++;
		}
		if(tmp.x<src.cols-1 && tmp.y<src.rows-1 && src.at<unsigned char>(tmp.y+1, tmp.x+1)==value){
			neiPoints.push_back(Point2i(tmp.x+1,tmp.y+1));
			src.at<unsigned char>(tmp.y+1, tmp.x+1) = label;//访问过
			top++;
		}
		if(tmp.x<src.cols-1 && tmp.y<src.rows-1 && src.at<unsigned char>(tmp.y+1, tmp.x+1)==value){
			neiPoints.push_back(Point2i(tmp.x+1,tmp.y+1));
			src.at<unsigned char>(tmp.y+1, tmp.x+1) = label;//访问过
			top++;
		}
		if(tmp.x<src.cols-1 && tmp.y>0 && src.at<unsigned char>(tmp.y-1, tmp.x+1)==value){
			neiPoints.push_back(Point2i(tmp.x+1,tmp.y-1));
			src.at<unsigned char>(tmp.y-1, tmp.x+1) = label;//访问过
			top++;
		}
		if(tmp.x>0 && tmp.y<src.rows-1 && src.at<unsigned char>(tmp.y+1, tmp.x-1)==value){
			neiPoints.push_back(Point2i(tmp.x-1,tmp.y+1));
			src.at<unsigned char>(tmp.y+1, tmp.x-1) = label;//访问过
			top++;
		}
		if(tmp.x>0 && tmp.y>0 && src.at<unsigned char>(tmp.y-1, tmp.x-1)==value){
			neiPoints.push_back(Point2i(tmp.x-1,tmp.y-1));
			src.at<unsigned char>(tmp.y-1, tmp.x-1) = label;//访问过
			top++;
		}
	}
}


void directBlender::prepare(vector<Mat> &warpSeamMask, vector<Mat> &warpMask, vector<Point> &topleft)
{
	//这里要求Mask都是CV_8UC1的类型
    int imgNum=warpMask.size();
    //首先利用tl来求最终的图像大小,这里要求tl均为正值
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
		seamMasks_.push_back(warpSeamMask[i].clone());//保存warpSeamMask供adjustForground使用
	}
//	foregroudMap=Mat::zeros(outRows,outCols,CV_8UC1);//初始化全局前景map
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
	panoImgOut=Mat::zeros(outRows,outCols,CV_8UC3);  //RGB格式,生成图像,这边要初始化为零啊啊啊啊兄弟！！！！！
	for (int i=0;i<imgNum;i++){
		Point &topLeft=topleft[i];
		Mat &imgNow = warpImg[i];
		Mat grayNow;
		cvtColor(warpImg[i], grayNow, CV_BGR2GRAY);
		for (int r=0;r<imgNow.rows;r++){
			unsigned char* grayOutSaved = grayImgOut_.ptr<unsigned char>(r+topLeft.y);
			unsigned char* grayImgSaved = grayImgs_[i].ptr<unsigned char>(r);
			unsigned char* foreOutMask = foreOut_.ptr<unsigned char>(r+topLeft.y);//存储着灰度值不能更新的蒙板
			unsigned char* foreImgMask = foreMasks_[i].ptr<unsigned char>(r);//存储着前景蒙板
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
					if(foreOutMask[c+xOF]){//不能更新
						if(foreImgMask[c]==0)//0保留 1抹掉
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
						//更新
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
	foreMasks_.clear();//每幅图中需要特殊处理的前景连通域  =1,抹掉 =0保留
	warpForeMasks_.clear();//保存warpForeMsk供doBlend使用
	foreOut_ = Mat::zeros(outRows,outCols,CV_8UC1);//全局需要特殊处理的前景连通域 =1 根据foreMask来处理 =0 不特殊处理
	vector<vector<Point2i>> vecNeiPoints;//保存穿过重叠区边缘的连通域
	vector<Point2i> neiPoints;
	vector<pair<int, int>> maskIndex;
	for (int i=0;i<imgNum;i++){
		foreMasks_.push_back(Mat::zeros(warpForeMsk[i].size(),warpForeMsk[i].type()));
		warpForeMasks_.push_back(warpForeMsk[i].clone());//保存原始的背景建模得到的前景mask，以后如果用不到要删掉？？？？
		if (warpForeMsk[i].size()!=warpImg[i].size()) throw sysException("Different size of warpedImg and warpedImgMask");
		if (warpForeMsk[i].type()!=CV_8UC1) throw sysException("Type of foreground mask matrix is worng! Should be CV_8UC1!");

		//二值化
		threshold(warpForeMsk[i],warpForeMsk[i],10.0f,255,0);
	}
	vecNeiPoints.clear();
	maskIndex.clear();
	for(int i=0;i<imgNum;i++){		
		neiPoints.clear();		
		Mat& fmsk = warpForeMsk[i];
		Mat& seamMask = seamMasks_[i];
		//寻找跨越重叠区边缘的连通域
		for (int r=0;r<fmsk.rows;++r){
			unsigned char *tptr=fmsk.ptr<unsigned char>(r);  //要生成的连通域模板
			unsigned char *seamMaskRow=seamMask.ptr<unsigned char>(r);
			for (int c=0;c<fmsk.cols;++c){
				if (tptr[c]==0xFF && seamMaskRow[c]>0){ //前景 跨过重叠区边界 //125这个标记值似乎没用了之后再删吧     注意这个地方是不对的 125应该在seammask里判断，先只是测试
					NeighbourSearch(fmsk,Point2i(c,r),neiPoints,vecNeiPoints.size()+1);
					//判断该连通域是否全在 树上较低图像 的重叠区内，即是否全是125
					int flag = -1;
					for(int ii=0;ii<neiPoints.size();ii++){
						int value = seamMask.at<unsigned char>(neiPoints[ii].y, neiPoints[ii].x);
						if(value>0&&value!=125){
							flag=255-value;
							break;
						}
					}
					if(flag==-1){//flag>0说明连通域对应的seamMask里有不是125的值，即不全在重叠区里,flag=-1就全在重叠区里，要抹掉
						for(int ii=0;ii<neiPoints.size();ii++){
							int y = neiPoints[ii].y;
							int x = neiPoints[ii].x;
							foreMasks_[i].at<unsigned char>(y, x) = 1;//抹掉
							foreOut_.at<unsigned char>(y+topleft[i].y, x+topleft[i].x)= 1;
						}
					}
					else{
						vector<Point2i> points = neiPoints;
						vecNeiPoints.push_back(points);
						if(flag>imgNum)throw sysException("flag");
						maskIndex.push_back(make_pair(i, flag));//flag与图像i对应的图像index,它们共同组成了这个重叠区
					}
				}
			}
		}
	}
	//寻找树上低的图像中的连通域 对应的 树上高的图像中的连通域
	for(int k=0;k<vecNeiPoints.size();k++){
		if(maskIndex[k].first!=2){//判断是否是树上低的
			int i = maskIndex[k].first;
			int index = maskIndex[k].second;
			//在对应图像上的映射
			int deltax = topleft[i].x-topleft[index].x;
			int deltay = topleft[i].y-topleft[index].y;
			vector<Point2i> mapPoints;
			Mat map = Mat::zeros(warpForeMsk[index].size(),warpForeMsk[index].type());//存储图像i的前景到图像index的映射						
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
				//在与之对应的图像中寻找具有重叠点数最多的连通域，且判断连通域是否穿过重叠区边缘
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
						//neiPoints与map重叠点数
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
				 if(overCnt>0){//在对应图像上找到了连通域
					if(maxOverIndex>0){//对应图像上连通域穿过重叠区边界，保留对应图像上的连通域
						//将该图像上的前景抹掉
						for(int ii=0;ii<vecNeiPoints[k].size();ii++){
							int y = vecNeiPoints[k][ii].y;
							int x = vecNeiPoints[k][ii].x;
							foreMasks_[i].at<unsigned char>(y, x) = 1;
							foreOut_.at<unsigned char>(y+topleft[i].y, x+topleft[i].x)= 1;							
						}
						//将对应图像前景设置为特殊处理，且找到对应图像在该图像上的映射
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
					else{////对应图像上连通域没有穿过重叠区边界，保留该图像上的连通域（树上较低的）
						//将对应图像上的连通域抹掉
						int width = warpForeMsk[i].cols,height = warpForeMsk[i].rows;
						for(int jj=0;jj<maxOverlap.size();jj++){
							int y = maxOverlap[jj].y, x= maxOverlap[jj].x;
							foreMasks_[index].at<unsigned char>(y, x) = 1;
							foreOut_.at<unsigned char>(y+topleft[index].y,x+topleft[index].x)=1;
							y -= deltay;
							x -= deltax;
							if((y>=0&&y<height)&&(x>=0&&x<width)){
								foreMasks_[i].at<unsigned char>(y, x) = 0;//防止出现空洞,两幅对应图像都抹掉的话，要恢复一幅														 
							}
						}						
					}
					
				 }
			
			}//end of if(mapPoints)
			//在对应图像上没有映射点 和 在对应图像上没找到连通域都这样处理
			for(int ii=0;ii<vecNeiPoints[k].size();ii++){
				int y = vecNeiPoints[k][ii].y;
				int x = vecNeiPoints[k][ii].x;
				//foreMasks_[i].at<unsigned char>(y, x) = 0;//在该图像上的保留
				foreOut_.at<unsigned char>(y+topleft[i].y, x+topleft[i].x)= 1;
			}//end of for(int ii = 0
			if(mapPoints.size()>0){//在对应图像上的映射抹掉
				for(int jj=0;jj<mapPoints.size();jj++){
					int y = mapPoints[jj].y, x= mapPoints[jj].x;
					foreMasks_[index].at<unsigned char>(y, x) = 1;
					foreOut_.at<unsigned char>(y+topleft[index].y,x+topleft[index].x)=1;	
				}
			}
		}//end of if(maskIndex)			
	}//end of for(int k=0)
}