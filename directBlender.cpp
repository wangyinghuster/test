#include "directBlender.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp> 


static void NeighbourSearch(Mat& src, Point2i floodPoint, vector<Point2i>& neiPoints)
{
	int value = src.at<unsigned char>(floodPoint.y, floodPoint.x);
	int bottom = 0,top=1;
	Point2i tmp;
	neiPoints.clear();
	neiPoints.push_back(floodPoint);
	src.at<unsigned char>(floodPoint.y, floodPoint.x) = 0;
	while(bottom!=top){
		tmp = neiPoints[bottom++];
		if(tmp.y>0&&src.at<unsigned char>(tmp.y-1, tmp.x)==value){
			neiPoints.push_back(Point2i(tmp.x,tmp.y-1));
			src.at<unsigned char>(tmp.y-1, tmp.x) = 0;//访问过
			top++;
		}
		if(tmp.y<src.rows-1 && src.at<unsigned char>(tmp.y+1, tmp.x)==value){
			neiPoints.push_back(Point2i(tmp.x,tmp.y+1));
			src.at<unsigned char>(tmp.y+1, tmp.x) = 0;//访问过
			top++;
		}
		if(tmp.x>0&&src.at<unsigned char>(tmp.y, tmp.x-1)==value){
			neiPoints.push_back(Point2i(tmp.x-1,tmp.y));
			src.at<unsigned char>(tmp.y, tmp.x-1) = 0;//访问过
			top++;
		}
		if(tmp.x<src.cols-1 && src.at<unsigned char>(tmp.y, tmp.x+1)==value){
			neiPoints.push_back(Point2i(tmp.x+1,tmp.y));
			src.at<unsigned char>(tmp.y, tmp.x+1) = 0;//访问过
			top++;
		}
		if(tmp.x<src.cols-1 && tmp.y<src.rows-1 && src.at<unsigned char>(tmp.y+1, tmp.x+1)==value){
			neiPoints.push_back(Point2i(tmp.x+1,tmp.y+1));
			src.at<unsigned char>(tmp.y+1, tmp.x+1) = 0;//访问过
			top++;
		}
		if(tmp.x<src.cols-1 && tmp.y<src.rows-1 && src.at<unsigned char>(tmp.y+1, tmp.x+1)==value){
			neiPoints.push_back(Point2i(tmp.x+1,tmp.y+1));
			src.at<unsigned char>(tmp.y+1, tmp.x+1) = 0;//访问过
			top++;
		}
		if(tmp.x<src.cols-1 && tmp.y<src.rows-1 && src.at<unsigned char>(tmp.y+1, tmp.x+1)==value){
			neiPoints.push_back(Point2i(tmp.x+1,tmp.y+1));
			src.at<unsigned char>(tmp.y+1, tmp.x+1) = 0;//访问过
			top++;
		}
		if(tmp.x<src.cols-1 && tmp.y>0 && src.at<unsigned char>(tmp.y-1, tmp.x+1)==value){
			neiPoints.push_back(Point2i(tmp.x+1,tmp.y-1));
			src.at<unsigned char>(tmp.y-1, tmp.x+1) = 0;//访问过
			top++;
		}
		if(tmp.x>0 && tmp.y<src.rows-1 && src.at<unsigned char>(tmp.y+1, tmp.x-1)==value){
			neiPoints.push_back(Point2i(tmp.x-1,tmp.y+1));
			src.at<unsigned char>(tmp.y+1, tmp.x-1) = 0;//访问过
			top++;
		}
		if(tmp.x>0 && tmp.y>0 && src.at<unsigned char>(tmp.y-1, tmp.x-1)==value){
			neiPoints.push_back(Point2i(tmp.x-1,tmp.y-1));
			src.at<unsigned char>(tmp.y-1, tmp.x-1) = 0;//访问过
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
	foreMasks_.clear();//保留穿过边缘的连通域
	warpForeMasks_.clear();//保存warpForeMsk供doBlend使用
	foreOut_ = Mat::zeros(outRows,outCols,CV_8UC1);
	vector<vector<Point2i>> vecNeiPoints;
	vector<Point2i> neiPoints;
	vector<pair<int, int>> maskIndex;
	for (int i=0;i<imgNum;i++){
		foreMasks_.push_back(Mat::zeros(warpForeMsk[i].size(),warpForeMsk[i].type()));
		warpForeMasks_.push_back(warpForeMsk[i].clone());
		if (warpForeMsk[i].size()!=warpImg[i].size()) throw sysException("Different size of warpedImg and warpedImgMask");
		if (warpForeMsk[i].type()!=CV_8UC1) throw sysException("Type of foreground mask matrix is worng! Should be CV_8UC1!");

		//二值化
		threshold(warpForeMsk[i],warpForeMsk[i],10.0f,255,0);
	}
	Mat temp;
	for(int i=0;i<imgNum;i++){
		vecNeiPoints.clear();
		neiPoints.clear();
		maskIndex.clear();
		Mat& fmsk = warpForeMsk[i];
		temp = fmsk.clone();
		Mat& seamMask = seamMasks_[i];
		//寻找跨越边缘的连通域
		for (int r=0;r<fmsk.rows;++r){
			unsigned char *tptr=temp.ptr<unsigned char>(r);  //要生成的连通域模板
			unsigned char *seamMaskRow=seamMask.ptr<unsigned char>(r);
			for (int c=0;c<fmsk.cols;++c){
				if (tptr[c]==0xFF && seamMaskRow[c]>0){ //前景 跨过重叠区边界 //125这个标记值似乎没用了之后再删吧     注意这个地方是不对的 125应该在seammask里判断，先只是测试
					NeighbourSearch(temp,Point2i(c,r),neiPoints);
					//判断该连通域是否全在 树上较低图像 的重叠区内，即是否全是125
					int flag = 1;
					for(int ii=0;ii<neiPoints.size();ii++)
						if(seamMask.at<unsigned char>(neiPoints[ii].y, neiPoints[ii].x)!=125){
							flag=0;break;
						}
					if(flag){//flag=0说明连通域对应的seamMask里有不是125的值，即不全在重叠区里,flag=1就全在重叠区里，要抹掉
						for(int ii=0;ii<neiPoints.size();ii++){
							int y = neiPoints[ii].y;
							int x = neiPoints[ii].x;
							foreMasks_[i].at<unsigned char>(y, x) = 1;
							foreOut_.at<unsigned char>(y+topleft[i].y, x+topleft[i].x)= 1;
						}
					}	
					vector<Point2i> points = neiPoints;
					vecNeiPoints.push_back(points);
					maskIndex.push_back(make_pair(i, 255-seamMaskRow[c]));//255-seamMaskRow[c]与图像i对应的图像index,它们共同组成了这个重叠区
				}
			}
		}
		temp.release();
	}
	//寻找树上低的图像中的连通域 对应的 树上高的图像中的连通域
	for(int k=0;k<vecNeiPoints.size();k++){
		if(maskIndex[k].first!=2){//判断是否是树上低的
			//遍历树上高的图像
		}			
	}

		for(int k=0;k<vecNeiPoints.size();k++){
			int index = maskIndex[k]; //与其重叠的图像索引
			int deltax = topleft[i].x-topleft[index].x;
			int deltay = topleft[i].y-topleft[index].y;
			vector<Point2i> mapPoints;
			vector<Point2i> maxOverlap;
			mapPoints.clear();
			int width = warpForeMsk[index].cols,height = warpForeMsk[index].rows;

			//Mat grayTest;
			//cvtColor(warpImg[i], grayTest, CV_BGR2GRAY);
			for(int j=0;j<vecNeiPoints[k].size();j++){
				int y = vecNeiPoints[k][j].y+deltay, x = vecNeiPoints[k][j].x+deltax;
				if((y>0&&y<height)&&(x>0&&x<width))
					mapPoints.push_back(Point2i(x,y));
				//warpForeMsk[i].at<unsigned char>(vecNeiPoints[k][j].y, vecNeiPoints[k][j].x) =  grayTest.at<unsigned char>(vecNeiPoints[k][j].y, vecNeiPoints[k][j].x);
			}
			if(mapPoints.size()>0){
				Mat map = Mat::zeros(warpForeMsk[index].size(),warpForeMsk[index].type());//存储图像i的前景到图像index的映射
				for(int n=0;n<mapPoints.size();n++){
					map.at<unsigned char>(mapPoints[n].y, mapPoints[n].x) = 255;
					foreMasks_[index].at<unsigned char>(mapPoints[n].y, mapPoints[n].x) = 1;//将前景在图像index的映射mask置为1，blend的时候该映射会被抹掉
				}
				int overCnt=0;
				temp = warpForeMsk[index].clone();
				//在warpForeMsk[index]里以(vecNeiPoints[k][j].y+deltay, vecNeiPoints[k][j].x+deltax)为种子点，寻找连通域
				//与map重叠最多的连通域认为是 在图像index中与vecNeiPoints[k]对应的连通域				
				for(int n=0;n<mapPoints.size();n++){
					int tmpCnt=0;
					neiPoints.clear();
					if(temp.at<unsigned char>(mapPoints[n].y, mapPoints[n].x)==0xff)
						NeighbourSearch(temp, Point2i(mapPoints[n].x,mapPoints[n].y), neiPoints);
					//neiPoints与map重叠点数
					for(int m=0;m<neiPoints.size();m++)
						if(map.at<unsigned char>(neiPoints[m].y, neiPoints[m].x))
							tmpCnt++;
					if(tmpCnt>overCnt){
						overCnt = tmpCnt;
						maxOverlap.clear();
						maxOverlap = neiPoints;
					}				
				}
				//看是不是要加一个阈值判断重叠点数，判断图像index中对应的前景区穿过图像index中的重叠边线了吗，来决定
				for(int n=0;n<vecNeiPoints[k].size();n++){//给foreOut_赋值
					int y = vecNeiPoints[k][n].y;
					int x = vecNeiPoints[k][n].x;
					foreOut_.at<unsigned char>(y+topleft[i].y, x+topleft[i].x)= 1;
				}
				for(int n=0;n<maxOverlap.size();n++){
					int y = maxOverlap[n].y;
					int x = maxOverlap[n].x;
					foreMasks_[index].at<unsigned char>(y, x) = 1;
					foreOut_.at<unsigned char>(y+topleft[index].y, x+topleft[index].x)= 1;
				}
			}
			else{
				//保留树上较低的
			}	
			//重叠区的去留
			//foreMasks_[i]

		}
		//for(int k=0;k<vecNeiPoints.size();k++)
		//	for(int j=0;j<vecNeiPoints[k].size();j++)
		//		fmsk.at<unsigned char>(vecNeiPoints[k][j].y, vecNeiPoints[k][j].x) = 125;	
	}
	
}