#include "weightFrameBlender.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>  
#include <set>
#include <map>
#include <vector>
#include <list>
#include "math.h"

weightFrameBlender::weightFrameBlender(int lNum, int sRadio)  //默认为5层,默认融合痕迹宽度为全图的1/20
{
    layerNum=lNum;
    seamRadio=sRadio;
}
/*
void weightFrameBlender::doBlend(vector<Mat> &warpImg,vector<Point> &topleft, Mat &panoImgOut)
{
    int imgNum=warpImg.size();

    if (!prepared) {
        throw sysException("Need to initialize before blend!");
    }

    panoImgOut=Mat::zeros(outRows,outCols,CV_8UC3);  //RGB格式,生成图像,这边要初始化为零啊啊啊啊兄弟！！！！！

    for (int i=0;i<imgNum;i++){
        Mat &weightMap=imgMaskWeight[i];
        Point &topLeft=topleft[i];
        Mat &imgNow=warpImg[i];

        for (int r=0;r<imgNow.rows;r++){
            const Point3_<unsigned char>* imgNowRow = imgNow.ptr<Point3_<unsigned char> >(r);
            float* maskWeight = weightMap.ptr<float>(r);
            Point3_<unsigned char>* outRow = panoImgOut.ptr<Point3_<unsigned char> >(r+topLeft.y);
                                                                                                                               
            int xOF=topLeft.x; 

            for (int c=0;c<imgNow.cols;c++){
                outRow[c+xOF].x += (unsigned char)( imgNowRow[c].x * maskWeight[c] );
                outRow[c+xOF].y += (unsigned char)( imgNowRow[c].y * maskWeight[c] );
                outRow[c+xOF].z += (unsigned char)( imgNowRow[c].z * maskWeight[c] );
            }
        }
    }
}*/

void weightFrameBlender::doBlend(vector<Mat> &warpImg,vector<Point> &topleft, Mat &panoImgOut)
{
	int imgNum=warpImg.size();

	if (!prepared) {
		throw sysException("Need to initialize before blend!");
	}

	panoImgOut=Mat::zeros(outRows,outCols,CV_8UC3);  //RGB格式,生成图像,这边要初始化为零啊啊啊啊兄弟！！！！！

	for (int i=0;i<imgNum;i++){
		Mat &weightMap=imgMaskWeight[i];
		Point &topLeft=topleft[i];
		Mat &imgNow=warpImg[i];

		for (int r=0;r<imgNow.rows;r++){
			const Point3_<unsigned char>* imgNowRow = imgNow.ptr<Point3_<unsigned char> >(r);
			float* maskWeight = weightMap.ptr<float>(r);
			Point3_<unsigned char>* outRow = panoImgOut.ptr<Point3_<unsigned char> >(r+topLeft.y);
			unsigned char *imgIdx=foregroudMap.ptr<unsigned char>(r+topLeft.y);

			int xOF=topLeft.x; 

			for (int c=0;c<imgNow.cols;c++){
				if (imgIdx[c+xOF]!=0){
					if (imgIdx[c+xOF]==(unsigned char)(i+1)){
						outRow[c+xOF].x =imgNowRow[c].x;
						outRow[c+xOF].y =imgNowRow[c].y;
						outRow[c+xOF].z =imgNowRow[c].z;
					}
				}else{
					outRow[c+xOF].x += (unsigned char)( imgNowRow[c].x * maskWeight[c] );
					outRow[c+xOF].y += (unsigned char)( imgNowRow[c].y * maskWeight[c] );
					outRow[c+xOF].z += (unsigned char)( imgNowRow[c].z * maskWeight[c] );
				}
				
			}
		}
	}
}

void weightFrameBlender::prepare(vector<Mat> &warpSeamMask, vector<Mat> &warpMask, vector<Point> &topleft)
{
    //这里要求Mask都是CV_8UC1的类型
    int imgNum=warpSeamMask.size();
	imgMaskWeight.clear();

    //首先利用tl来求最终的图像大小,这里要求tl均为正值

    getAllSize(warpSeamMask,topleft);
    //先腐蚀边缘然后再用膨胀来生成加权的图像蒙版 图像蒙版32位
	//--__debug(cout <<"Row:"<<outRows<<" Col:"<<outCols<<endl;)

    //膨胀核函数

    Scalar color(255,255,255);
    Mat dilateKernel;
    int a=(int)(sqrt((double)outCols*outRows)/(2 * layerNum * seamRadio));             //这个核的值应该和全图的大小成比例对应,seamRadio是比例
    Mat paint=Mat::zeros(2*a+1,2*a+1,CV_8UC3);
    Point cnt1(a+1,a+1);
    circle(paint,cnt1,a,color,CV_FILLED);   //生成球形模板
    cvtColor(paint,dilateKernel,CV_RGB2GRAY);  //膨胀处理核的生成

    //腐蚀核函数
    Mat erodeKernel=dilateKernel; //腐蚀处理核的生成


    for (int i=0;i<imgNum;i++){
       vector<Mat> temp;
       Mat mask=warpSeamMask[i].clone();   //这个mask要进过处理的，所以用clone吧，别搞乱了。
       Mat &maskAll=warpMask[i];
       if ((maskAll.type()!=CV_8UC1)&&(mask.type()!=CV_8UC1)) throw sysException("Mask Matrix should be CV_8UC1!");
       Mat mashLayer;

       Mat weightFloat;

       //首先腐蚀maskAll几轮，腐蚀完毕在与seamMask相与，这样
       Mat maskTemp;
       erode(maskAll,maskTemp,erodeKernel,Point(-1,-1),layerNum+1);  //腐蚀layerNum+1次
       mask= mask & maskTemp;

       mask.convertTo(weightFloat, CV_32F, 1./255.);  //要缩小255倍
       temp.push_back(weightFloat.clone());   //保存原始mask的这个float的版本

       for (int k=0;k<layerNum;k++){   //先生成一个逐渐变大的mask

           dilate(mask,mashLayer,dilateKernel); // 膨胀

           mashLayer = mashLayer & maskAll ;  //用与函数消除边缘的膨胀

           mashLayer.convertTo(weightFloat, CV_32F, 1./255.);  //要缩小255倍
           temp.push_back(weightFloat.clone());   //保存这个float的版本

           mask=mashLayer;
       }

       Mat weightMap;
       weightMap.create(warpSeamMask[i].rows,warpSeamMask[i].cols,CV_32F);  //这个用32位float的矩阵，以表示精确
       weightMap.setTo(0);

       for (int k=0;k<layerNum+1;k++){

           Mat &maskNow=temp[k];
           for (int r=0;r<weightMap.rows;r++){   //这个for循环实际上是求和，但是不知道如果用openCV的函数代替会怎么样
               const float* maskNowRow = maskNow.ptr<float>(r);
               float* maskWeight = weightMap.ptr<float>(r);

               for (int c=0;c<weightMap.cols;c++){
                   maskWeight[c]+= maskNowRow[c];
               }
           }
       }

       for (int r=0;r<weightMap.rows;r++){   //这个for循环实际上是求除法，但是不知道如果用openCV的函数代替会怎么样
           float* maskWeight = weightMap.ptr<float>(r);

           for (int c=0;c<weightMap.cols;c++){
               maskWeight[c] = maskWeight[c]/(layerNum+1);
           }
       }

       //再进行一次高斯模糊
       Mat weightANS;
       GaussianBlur(weightMap,weightANS, Size(2*a+1, 2*a+1), 0, 0);
       weightANS.clone().copyTo(weightANS,maskAll) ;  //除去对边的模糊

       imgMaskWeight.push_back(weightANS);
    }

    //对所有生成的weight图像进行归一化
    //首先求和
    Mat weightMapAll;
    weightMapAll.create(outRows,outCols,CV_32F);  //全图大小
    weightMapAll.setTo(0);   //对！！这里要初始化为0啊兄弟！！

    for (int i=0;i<imgNum;i++){
        Mat &weightMap=imgMaskWeight[i];
        Point &topLeft=topleft[i];

        for (int r=0;r<weightMap.rows;r++){

            const float* maskWeight = weightMap.ptr<float>(r);
            float* allRow = weightMapAll.ptr<float>(r+topLeft.y);

            int xOF=topLeft.x;

            for (int c=0;c<weightMap.cols;c++){
                allRow[c+xOF] += maskWeight[c];
            }

        }
    }

    //再归一化

    for (int i=0;i<imgNum;i++){
        Mat &weightMap=imgMaskWeight[i];
        Point &topLeft=topleft[i];

        for (int r=0;r<weightMap.rows;r++){

            float* maskWeight = weightMap.ptr<float>(r);
            const float* allRow = weightMapAll.ptr<float>(r+topLeft.y);

            int xOF=topLeft.x;

            for (int c=0;c<weightMap.cols;c++){
                maskWeight[c] =  maskWeight[c]/allRow[c+xOF];
            }

        }

#ifdef IMGSHOW   //输出最终的blend mask
       ostringstream s1;
       s1 << i;
       cvNamedWindow(string("blend mask "+ s1.str()).c_str(), CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
       Mat weightTemp;
       weightMap.convertTo(weightTemp,CV_8UC1,255);
       cvShowImage(string("blend mask "+ s1.str()).c_str(),& IplImage(weightTemp));
       waitKey(30);
       cvSaveImage(string("blend_mask_"+ s1.str()+".jpg").c_str(),& IplImage(weightTemp));
#endif
    }

	foregroudMap=Mat::zeros(outRows,outCols,CV_8UC1);//初始化全局前景map
    prepared=true;
}

static double pointDst(Point2f & p1,Point2f & p2){
	return sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y));
}

//这里需要重载point2i的比较函数
//map的比较函数
struct point2icmp
{
	bool operator ()(const Point2i &p1,const Point2i &p2) const
	{
		if (p1.x!=p2.x)
			return p1.x<p2.x;
		else
			return p1.y<p2.y;
	}
};



float weightFrameBlender::destTher=30.0f;  //前景判别的距离阈值

void weightFrameBlender::adjustForground(vector<Mat> &warpImg,vector<Mat> warpForeMsk,vector<Point> topleft)
{
	//默认输入的warpForeMsk是二值化的,并且要求warpForeMask的格式是CV_8UC1;
	//连通域检测

	
	vector<vector<Point2f > > centerSet;//保存的前景区域的中心值
	vector<vector<Point2i> > floodSeed;//漫水算法的起点值
	vector<vector<int> > foreSize; //保存的前景区域中心的大小
	overMask.clear();

	int numImg=(int)warpImg.size();

	//生成跨越边界的前景图像模板
	for (int i=0;i<numImg;++i){
		
		Mat &fmsk=warpForeMsk[i];  //前景模板图像
		if (fmsk.size()!=warpImg[i].size()) throw sysException("Different size of warpedImg and warpedImgMask");


		Mat &weightMap=imgMaskWeight[i]; //权值图像

		Mat temp=Mat::zeros(fmsk.rows,fmsk.cols,CV_8UC1);  //保存临时图像
		vector<int> fsz;  //保存连通域的size
		vector<Point2i> fseed; //保存漫水的起始点 

		unsigned char cflag=254; //连通域标志

		if (fmsk.type()!=CV_8UC1) throw sysException("Type of foreground mask matrix is worng! Should be CV_8UC1!");

		//二值化
		threshold(fmsk,temp,10.0f,255,0);
		//寻找跨越拼接缝的连通域
		for (int r=0;r<fmsk.rows;++r){
			//unsigned char *fptr=fmsk.ptr<unsigned char>(r);  //这个mask是0和255
			float* mwptr = weightMap.ptr<float>(r);  //这个mask是0.0到1.0
			unsigned char *tptr=temp.ptr<unsigned char>(r);  //要生成的连通域模板

			for (int c=0;c<fmsk.cols;++c){
				if ((tptr[c]==0xff)&&(mwptr[c]>0.1)&&(mwptr[c]<0.9)){  //表示为255
					 floodFill(temp,Point(c,r),cflag--); //漫水算法
					 fsz.push_back(0);
					 fseed.push_back(Point2i(c+topleft[i].x,r+topleft[i].y)); //保存漫水的起始点，不过这个要加上偏移
				}
			}
		}

	/*	__debug(
		ostringstream s1;
		s1 << i;
		cvNamedWindow(string("overMask "+ s1.str()).c_str(),CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
		cvShowImage(string("overMask "+ s1.str()).c_str(),& IplImage(temp));)*/

		//计算那些连通域大小和中心点
		vector<Point2f > cent(fsz.size(),Point2f(0.0,0.0));
		for (int r=0;r<fmsk.rows;++r){
			unsigned char *tptr=temp.ptr<unsigned char>(r);  //这个mask的值从
			for (int c=0;c<fmsk.cols;++c){
				if ((tptr[c])&&(tptr[c]!=0xff)){  //表示该处有前景
					int fi =254-tptr[c]; //计算对应的前景标号
					++fsz[fi];     //前景大小加一
					cent[fi].x+=c; //中心点坐标更新
					cent[fi].y+=r;
				}
			}
		}
		for (unsigned int ti=0;ti<fsz.size();++ti){  //计算中心值，并且加上偏移
			cent[ti].x=cent[ti].x/fsz[ti]+topleft[i].x;
			cent[ti].y=cent[ti].y/fsz[ti]+topleft[i].y;
		}
	
		overMask.push_back(temp); //保存前景蒙版
		centerSet.push_back(cent);//保存的前景区域的中心值
		foreSize.push_back(fsz); //保存的前景区域中心的大小
		floodSeed.push_back(fseed); //保存漫水的起始点
	}

	//我突然觉得这里就用一些阈值方法来判断目标的匹配关系了，不用isodata这种杀器，之后再说。
    //这里下面的Point2i 用作索引，x为图像号，y为对应的连通域号。
	//且没有考虑虽然连通但是没有被判断为同一个目标 ,这里这个是问题
	//或者不联通了却被判断成同一个目标
	map<Point2i,list<vector<Point2i > >::iterator,point2icmp> revMap; //关联映射 点索引集合->类指针
	list< vector<Point2i > > clsMap;  //类标号 -> 类集合内的点的索引内容
	for (unsigned int i=0;i<centerSet.size();++i){
		vector<Point2f > &cent = centerSet[i];
		for (unsigned int j=0;j<cent.size();++j){   //暴力搜索	
			Point2i ct(i,j);  //记录点的索引

			for (unsigned int fki=i+1;fki<centerSet.size();++fki){
				vector<Point2f > &centQuer=centerSet[fki];
				for (unsigned int fkj=0;fkj<centQuer.size();++fkj){
					if (pointDst(cent[j],centQuer[fkj])<destTher)  //距离小于destTher个像素
					{
						
						Point2i ctQ(fki,fkj); //记录点的索引
						if (revMap.count(ct)==0){
							if (revMap.count(ctQ)==0){  //这两个都没有集合
								vector<Point2i> ptmp;
								ptmp.push_back(ct);
								ptmp.push_back(ctQ);
								clsMap.push_front(ptmp);
								list<vector<Point2i> >::iterator &ptp=clsMap.begin();
								revMap[ctQ]=ptp;
								revMap[ct]=ptp;
							}else {  //第二个有集合
							    revMap[ct]=revMap[ctQ];
								list<vector<Point2i> >::iterator &ptp=revMap[ct];
							    ptp->push_back(ct);
							}
						}else {
							if (revMap.count(ctQ)==0){  //第一个有第二个没有
							    revMap[ctQ]=revMap[ct];
								list<vector<Point2i> >::iterator &ptp=revMap[ctQ];
								ptp->push_back(ctQ);
 							}else {  //两个都有集合,且集合不应该相等,合并集合
								list<vector<Point2i> >::iterator ptpSrc=revMap[ct]; //这里不能用别名
								list<vector<Point2i> >::iterator ptpDst=revMap[ctQ];
								
								if ((ptpSrc)!=(ptpDst)) {
									for (unsigned int u=0;u<(ptpSrc->size());++u){ //将src集合的东西全部转入dst
										revMap[(*ptpSrc)[u]]=ptpDst; //这里会修改别名
										ptpDst->push_back((*ptpSrc)[u]);
									}
									clsMap.erase(ptpSrc);  //删除这个集合
								}
							}
						}  
					}
				}
			}

			if (revMap.count(ct)==0){  //暴力搜索完成都没有找到一个匹配的，那么自成一个块
				vector<Point2i> ptmp;
				ptmp.push_back(ct);
				clsMap.push_front(ptmp);
				list<vector<Point2i> >::iterator &ptp=clsMap.begin();
				revMap[ct]=ptp;
			}
		}
	}

	//cout << "find " << clsMap.size() <<" Object!"<<endl;

	//对前景mask坐上加权，和测量
	//这里直接生成一张全图，填充各个连通区域，连通区域的值，作为图像像素的索引符号
	Mat lastForeMap=foregroudMap;
	foregroudMap=Mat::zeros(outRows,outCols,CV_8UC1);
	//生成一张只有跨越拼接缝边缘的连通区域的图
	for (unsigned int k=0;k<overMask.size();++k){
		Mat &tp=foregroudMap(Rect(topleft[k].x,topleft[k].y,overMask[k].cols,overMask[k].rows));
		Mat &msk=overMask[k];
		for (int r=0;r<msk.rows;++r){
			unsigned char *fmap=tp.ptr<unsigned char>(r);
			unsigned char *ptmsk=msk.ptr<unsigned char>(r);
			for (int c=0;c<msk.cols;++c){
				if ((ptmsk[c])&&(ptmsk[c]!=0xff)){
					fmap[c]=0xff;
				}
			}
		}
	}


	//从上一帧的前景图像得到继承的值
	//如果这一帧的mask和上一帧的mask有重叠的话
	//那么这一帧的mask继承上一帧的图像标号
	for (int r=0;r<outRows;++r){
		unsigned char *fmap=foregroudMap.ptr<unsigned char>(r);
		unsigned char *lfmp=lastForeMap.ptr<unsigned char>(r);
		for (int c=0;c<outCols;++c){		
			if ((fmap[c]==0xff)&&(lfmp[c])&&(lfmp[c]!=0xff)){  //这里不知道为什么不能少最后一个条件，这里是个问题
				floodFill(foregroudMap,Point(c,r),lfmp[c]); //漫水为上一帧图像的索引值
			}
		}
	}


/*
	__debug(namedWindow("foregroundMask",CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);)
	__debug(cvShowImage("foregroundMask",& IplImage(foregroudMap));)
	__debug(waitKey(50);)*/

    //对所有判别为同一个类的目标寻找一个面积最大的前景值 //这个选择方案没有考虑到会有边缘存在的问题,且
	//且没有考虑虽然连通但是没有被判断为同一个目标
	//或者不联通了却被判断成同一个目标
	//由于以上问题这里需要用到排序的想法
	vector<Point2i> miVec;
	vector<int> sizVec;
	vector<int> sdSiz;

	for (list<vector<Point2i > >::iterator it=clsMap.begin();it!=clsMap.end();++it){
		vector<Point2i > &ptSet=*it;
		//寻找一个size的最大值
		int maxIdx=0;
		int maxSize=-1;
		int secMaxSz=-1; //代表未初始化，只有一个值，没有第二大
		for (unsigned int k=0;k<ptSet.size();++k){
			if (foreSize[ptSet[k].x][ptSet[k].y]>maxSize){
				maxIdx=k;
				secMaxSz=maxSize;
				maxSize=foreSize[ptSet[k].x][ptSet[k].y];
			}else if (foreSize[ptSet[k].x][ptSet[k].y]>secMaxSz){
				secMaxSz=foreSize[ptSet[k].x][ptSet[k].y];
			}
		}
		//插入排序
		sizVec.push_back(maxSize);
		miVec.push_back(ptSet[maxIdx]);
		sdSiz.push_back(secMaxSz);

		int ik=(int)sizVec.size()-2;
		while((ik>=0)&&(sizVec[ik])>maxSize){
			sizVec[ik+1]=sizVec[ik];
			miVec[ik+1]=miVec[ik];
			sdSiz[ik+1]=sdSiz[ik];
			ik--;
		}

		sizVec[ik+1]=maxSize;
		miVec[ik+1]=ptSet[maxIdx];
		sdSiz[ik+1]=secMaxSz;
		
	}

    int j=0;
	for (vector<Point2i>::iterator it=miVec.begin();it!=miVec.end();++it,j++){
		cout << sizVec[j] <<","<< sdSiz[j] <<";";
		if ((sdSiz[j]==-1)                     //第一大为唯一的值，或者第一大比第二大差太多才更新新的漫水图像。
			||(sizVec[j]>sdSiz[j]+5500)        //类似于施密特触发器的延迟触发设置
			||(foregroudMap.at<unsigned char>(floodSeed[it->x][it->y].y,floodSeed[it->x][it->y].x)==0xff)) //或者这部分还是255
													   
		{
			//这里漫水起始点不设为中心点是因为可能中心点不在前景区内部
			floodFill(foregroudMap,floodSeed[it->x][it->y],(it->x)+1); //漫水为图像的索引值加一
		}
	}
	
	/*__debug(Mat temp;)
	__debug(foregroudMap.convertTo(temp,foregroudMap.type(),80.0);)
	__debug(namedWindow("floodFillMask",CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);)
	__debug(cvShowImage("floodFillMask",& IplImage(temp));)*/
	//__debug(waitKey(0);)/**/

}


