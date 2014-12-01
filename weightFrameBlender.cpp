#include "weightFrameBlender.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>  
#include <set>
#include <map>
#include <vector>
#include <list>
#include "math.h"

weightFrameBlender::weightFrameBlender(int lNum, int sRadio)  //Ĭ��Ϊ5��,Ĭ���ںϺۼ����Ϊȫͼ��1/20
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

    panoImgOut=Mat::zeros(outRows,outCols,CV_8UC3);  //RGB��ʽ,����ͼ��,���Ҫ��ʼ��Ϊ�㰡�������ֵܣ���������

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

	panoImgOut=Mat::zeros(outRows,outCols,CV_8UC3);  //RGB��ʽ,����ͼ��,���Ҫ��ʼ��Ϊ�㰡�������ֵܣ���������

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
    //����Ҫ��Mask����CV_8UC1������
    int imgNum=warpSeamMask.size();
	imgMaskWeight.clear();

    //��������tl�������յ�ͼ���С,����Ҫ��tl��Ϊ��ֵ

    getAllSize(warpSeamMask,topleft);
    //�ȸ�ʴ��ԵȻ���������������ɼ�Ȩ��ͼ���ɰ� ͼ���ɰ�32λ
	//--__debug(cout <<"Row:"<<outRows<<" Col:"<<outCols<<endl;)

    //���ͺ˺���

    Scalar color(255,255,255);
    Mat dilateKernel;
    int a=(int)(sqrt((double)outCols*outRows)/(2 * layerNum * seamRadio));             //����˵�ֵӦ�ú�ȫͼ�Ĵ�С�ɱ�����Ӧ,seamRadio�Ǳ���
    Mat paint=Mat::zeros(2*a+1,2*a+1,CV_8UC3);
    Point cnt1(a+1,a+1);
    circle(paint,cnt1,a,color,CV_FILLED);   //��������ģ��
    cvtColor(paint,dilateKernel,CV_RGB2GRAY);  //���ʹ���˵�����

    //��ʴ�˺���
    Mat erodeKernel=dilateKernel; //��ʴ����˵�����


    for (int i=0;i<imgNum;i++){
       vector<Mat> temp;
       Mat mask=warpSeamMask[i].clone();   //���maskҪ��������ģ�������clone�ɣ�������ˡ�
       Mat &maskAll=warpMask[i];
       if ((maskAll.type()!=CV_8UC1)&&(mask.type()!=CV_8UC1)) throw sysException("Mask Matrix should be CV_8UC1!");
       Mat mashLayer;

       Mat weightFloat;

       //���ȸ�ʴmaskAll���֣���ʴ�������seamMask���룬����
       Mat maskTemp;
       erode(maskAll,maskTemp,erodeKernel,Point(-1,-1),layerNum+1);  //��ʴlayerNum+1��
       mask= mask & maskTemp;

       mask.convertTo(weightFloat, CV_32F, 1./255.);  //Ҫ��С255��
       temp.push_back(weightFloat.clone());   //����ԭʼmask�����float�İ汾

       for (int k=0;k<layerNum;k++){   //������һ���𽥱���mask

           dilate(mask,mashLayer,dilateKernel); // ����

           mashLayer = mashLayer & maskAll ;  //���뺯��������Ե������

           mashLayer.convertTo(weightFloat, CV_32F, 1./255.);  //Ҫ��С255��
           temp.push_back(weightFloat.clone());   //�������float�İ汾

           mask=mashLayer;
       }

       Mat weightMap;
       weightMap.create(warpSeamMask[i].rows,warpSeamMask[i].cols,CV_32F);  //�����32λfloat�ľ����Ա�ʾ��ȷ
       weightMap.setTo(0);

       for (int k=0;k<layerNum+1;k++){

           Mat &maskNow=temp[k];
           for (int r=0;r<weightMap.rows;r++){   //���forѭ��ʵ��������ͣ����ǲ�֪�������openCV�ĺ����������ô��
               const float* maskNowRow = maskNow.ptr<float>(r);
               float* maskWeight = weightMap.ptr<float>(r);

               for (int c=0;c<weightMap.cols;c++){
                   maskWeight[c]+= maskNowRow[c];
               }
           }
       }

       for (int r=0;r<weightMap.rows;r++){   //���forѭ��ʵ����������������ǲ�֪�������openCV�ĺ����������ô��
           float* maskWeight = weightMap.ptr<float>(r);

           for (int c=0;c<weightMap.cols;c++){
               maskWeight[c] = maskWeight[c]/(layerNum+1);
           }
       }

       //�ٽ���һ�θ�˹ģ��
       Mat weightANS;
       GaussianBlur(weightMap,weightANS, Size(2*a+1, 2*a+1), 0, 0);
       weightANS.clone().copyTo(weightANS,maskAll) ;  //��ȥ�Աߵ�ģ��

       imgMaskWeight.push_back(weightANS);
    }

    //���������ɵ�weightͼ����й�һ��
    //�������
    Mat weightMapAll;
    weightMapAll.create(outRows,outCols,CV_32F);  //ȫͼ��С
    weightMapAll.setTo(0);   //�ԣ�������Ҫ��ʼ��Ϊ0���ֵܣ���

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

    //�ٹ�һ��

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

#ifdef IMGSHOW   //������յ�blend mask
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

	foregroudMap=Mat::zeros(outRows,outCols,CV_8UC1);//��ʼ��ȫ��ǰ��map
    prepared=true;
}

static double pointDst(Point2f & p1,Point2f & p2){
	return sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y));
}

//������Ҫ����point2i�ıȽϺ���
//map�ıȽϺ���
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



float weightFrameBlender::destTher=30.0f;  //ǰ���б�ľ�����ֵ

void weightFrameBlender::adjustForground(vector<Mat> &warpImg,vector<Mat> warpForeMsk,vector<Point> topleft)
{
	//Ĭ�������warpForeMsk�Ƕ�ֵ����,����Ҫ��warpForeMask�ĸ�ʽ��CV_8UC1;
	//��ͨ����

	
	vector<vector<Point2f > > centerSet;//�����ǰ�����������ֵ
	vector<vector<Point2i> > floodSeed;//��ˮ�㷨�����ֵ
	vector<vector<int> > foreSize; //�����ǰ���������ĵĴ�С
	overMask.clear();

	int numImg=(int)warpImg.size();

	//���ɿ�Խ�߽��ǰ��ͼ��ģ��
	for (int i=0;i<numImg;++i){
		
		Mat &fmsk=warpForeMsk[i];  //ǰ��ģ��ͼ��
		if (fmsk.size()!=warpImg[i].size()) throw sysException("Different size of warpedImg and warpedImgMask");


		Mat &weightMap=imgMaskWeight[i]; //Ȩֵͼ��

		Mat temp=Mat::zeros(fmsk.rows,fmsk.cols,CV_8UC1);  //������ʱͼ��
		vector<int> fsz;  //������ͨ���size
		vector<Point2i> fseed; //������ˮ����ʼ�� 

		unsigned char cflag=254; //��ͨ���־

		if (fmsk.type()!=CV_8UC1) throw sysException("Type of foreground mask matrix is worng! Should be CV_8UC1!");

		//��ֵ��
		threshold(fmsk,temp,10.0f,255,0);
		//Ѱ�ҿ�Խƴ�ӷ����ͨ��
		for (int r=0;r<fmsk.rows;++r){
			//unsigned char *fptr=fmsk.ptr<unsigned char>(r);  //���mask��0��255
			float* mwptr = weightMap.ptr<float>(r);  //���mask��0.0��1.0
			unsigned char *tptr=temp.ptr<unsigned char>(r);  //Ҫ���ɵ���ͨ��ģ��

			for (int c=0;c<fmsk.cols;++c){
				if ((tptr[c]==0xff)&&(mwptr[c]>0.1)&&(mwptr[c]<0.9)){  //��ʾΪ255
					 floodFill(temp,Point(c,r),cflag--); //��ˮ�㷨
					 fsz.push_back(0);
					 fseed.push_back(Point2i(c+topleft[i].x,r+topleft[i].y)); //������ˮ����ʼ�㣬�������Ҫ����ƫ��
				}
			}
		}

	/*	__debug(
		ostringstream s1;
		s1 << i;
		cvNamedWindow(string("overMask "+ s1.str()).c_str(),CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
		cvShowImage(string("overMask "+ s1.str()).c_str(),& IplImage(temp));)*/

		//������Щ��ͨ���С�����ĵ�
		vector<Point2f > cent(fsz.size(),Point2f(0.0,0.0));
		for (int r=0;r<fmsk.rows;++r){
			unsigned char *tptr=temp.ptr<unsigned char>(r);  //���mask��ֵ��
			for (int c=0;c<fmsk.cols;++c){
				if ((tptr[c])&&(tptr[c]!=0xff)){  //��ʾ�ô���ǰ��
					int fi =254-tptr[c]; //�����Ӧ��ǰ�����
					++fsz[fi];     //ǰ����С��һ
					cent[fi].x+=c; //���ĵ��������
					cent[fi].y+=r;
				}
			}
		}
		for (unsigned int ti=0;ti<fsz.size();++ti){  //��������ֵ�����Ҽ���ƫ��
			cent[ti].x=cent[ti].x/fsz[ti]+topleft[i].x;
			cent[ti].y=cent[ti].y/fsz[ti]+topleft[i].y;
		}
	
		overMask.push_back(temp); //����ǰ���ɰ�
		centerSet.push_back(cent);//�����ǰ�����������ֵ
		foreSize.push_back(fsz); //�����ǰ���������ĵĴ�С
		floodSeed.push_back(fseed); //������ˮ����ʼ��
	}

	//��ͻȻ�����������һЩ��ֵ�������ж�Ŀ���ƥ���ϵ�ˣ�����isodata����ɱ����֮����˵��
    //���������Point2i ����������xΪͼ��ţ�yΪ��Ӧ����ͨ��š�
	//��û�п�����Ȼ��ͨ����û�б��ж�Ϊͬһ��Ŀ�� ,�������������
	//���߲���ͨ��ȴ���жϳ�ͬһ��Ŀ��
	map<Point2i,list<vector<Point2i > >::iterator,point2icmp> revMap; //����ӳ�� ����������->��ָ��
	list< vector<Point2i > > clsMap;  //���� -> �༯���ڵĵ����������
	for (unsigned int i=0;i<centerSet.size();++i){
		vector<Point2f > &cent = centerSet[i];
		for (unsigned int j=0;j<cent.size();++j){   //��������	
			Point2i ct(i,j);  //��¼�������

			for (unsigned int fki=i+1;fki<centerSet.size();++fki){
				vector<Point2f > &centQuer=centerSet[fki];
				for (unsigned int fkj=0;fkj<centQuer.size();++fkj){
					if (pointDst(cent[j],centQuer[fkj])<destTher)  //����С��destTher������
					{
						
						Point2i ctQ(fki,fkj); //��¼�������
						if (revMap.count(ct)==0){
							if (revMap.count(ctQ)==0){  //��������û�м���
								vector<Point2i> ptmp;
								ptmp.push_back(ct);
								ptmp.push_back(ctQ);
								clsMap.push_front(ptmp);
								list<vector<Point2i> >::iterator &ptp=clsMap.begin();
								revMap[ctQ]=ptp;
								revMap[ct]=ptp;
							}else {  //�ڶ����м���
							    revMap[ct]=revMap[ctQ];
								list<vector<Point2i> >::iterator &ptp=revMap[ct];
							    ptp->push_back(ct);
							}
						}else {
							if (revMap.count(ctQ)==0){  //��һ���еڶ���û��
							    revMap[ctQ]=revMap[ct];
								list<vector<Point2i> >::iterator &ptp=revMap[ctQ];
								ptp->push_back(ctQ);
 							}else {  //�������м���,�Ҽ��ϲ�Ӧ�����,�ϲ�����
								list<vector<Point2i> >::iterator ptpSrc=revMap[ct]; //���ﲻ���ñ���
								list<vector<Point2i> >::iterator ptpDst=revMap[ctQ];
								
								if ((ptpSrc)!=(ptpDst)) {
									for (unsigned int u=0;u<(ptpSrc->size());++u){ //��src���ϵĶ���ȫ��ת��dst
										revMap[(*ptpSrc)[u]]=ptpDst; //������޸ı���
										ptpDst->push_back((*ptpSrc)[u]);
									}
									clsMap.erase(ptpSrc);  //ɾ���������
								}
							}
						}  
					}
				}
			}

			if (revMap.count(ct)==0){  //����������ɶ�û���ҵ�һ��ƥ��ģ���ô�Գ�һ����
				vector<Point2i> ptmp;
				ptmp.push_back(ct);
				clsMap.push_front(ptmp);
				list<vector<Point2i> >::iterator &ptp=clsMap.begin();
				revMap[ct]=ptp;
			}
		}
	}

	//cout << "find " << clsMap.size() <<" Object!"<<endl;

	//��ǰ��mask���ϼ�Ȩ���Ͳ���
	//����ֱ������һ��ȫͼ����������ͨ������ͨ�����ֵ����Ϊͼ�����ص���������
	Mat lastForeMap=foregroudMap;
	foregroudMap=Mat::zeros(outRows,outCols,CV_8UC1);
	//����һ��ֻ�п�Խƴ�ӷ��Ե����ͨ�����ͼ
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


	//����һ֡��ǰ��ͼ��õ��̳е�ֵ
	//�����һ֡��mask����һ֡��mask���ص��Ļ�
	//��ô��һ֡��mask�̳���һ֡��ͼ����
	for (int r=0;r<outRows;++r){
		unsigned char *fmap=foregroudMap.ptr<unsigned char>(r);
		unsigned char *lfmp=lastForeMap.ptr<unsigned char>(r);
		for (int c=0;c<outCols;++c){		
			if ((fmap[c]==0xff)&&(lfmp[c])&&(lfmp[c]!=0xff)){  //���ﲻ֪��Ϊʲô���������һ�������������Ǹ�����
				floodFill(foregroudMap,Point(c,r),lfmp[c]); //��ˮΪ��һ֡ͼ�������ֵ
			}
		}
	}


/*
	__debug(namedWindow("foregroundMask",CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);)
	__debug(cvShowImage("foregroundMask",& IplImage(foregroudMap));)
	__debug(waitKey(50);)*/

    //�������б�Ϊͬһ�����Ŀ��Ѱ��һ���������ǰ��ֵ //���ѡ�񷽰�û�п��ǵ����б�Ե���ڵ�����,��
	//��û�п�����Ȼ��ͨ����û�б��ж�Ϊͬһ��Ŀ��
	//���߲���ͨ��ȴ���жϳ�ͬһ��Ŀ��
	//������������������Ҫ�õ�������뷨
	vector<Point2i> miVec;
	vector<int> sizVec;
	vector<int> sdSiz;

	for (list<vector<Point2i > >::iterator it=clsMap.begin();it!=clsMap.end();++it){
		vector<Point2i > &ptSet=*it;
		//Ѱ��һ��size�����ֵ
		int maxIdx=0;
		int maxSize=-1;
		int secMaxSz=-1; //����δ��ʼ����ֻ��һ��ֵ��û�еڶ���
		for (unsigned int k=0;k<ptSet.size();++k){
			if (foreSize[ptSet[k].x][ptSet[k].y]>maxSize){
				maxIdx=k;
				secMaxSz=maxSize;
				maxSize=foreSize[ptSet[k].x][ptSet[k].y];
			}else if (foreSize[ptSet[k].x][ptSet[k].y]>secMaxSz){
				secMaxSz=foreSize[ptSet[k].x][ptSet[k].y];
			}
		}
		//��������
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
		if ((sdSiz[j]==-1)                     //��һ��ΪΨһ��ֵ�����ߵ�һ��ȵڶ����̫��Ÿ����µ���ˮͼ��
			||(sizVec[j]>sdSiz[j]+5500)        //������ʩ���ش��������ӳٴ�������
			||(foregroudMap.at<unsigned char>(floodSeed[it->x][it->y].y,floodSeed[it->x][it->y].x)==0xff)) //�����ⲿ�ֻ���255
													   
		{
			//������ˮ��ʼ�㲻��Ϊ���ĵ�����Ϊ�������ĵ㲻��ǰ�����ڲ�
			floodFill(foregroudMap,floodSeed[it->x][it->y],(it->x)+1); //��ˮΪͼ�������ֵ��һ
		}
	}
	
	/*__debug(Mat temp;)
	__debug(foregroudMap.convertTo(temp,foregroudMap.type(),80.0);)
	__debug(namedWindow("floodFillMask",CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);)
	__debug(cvShowImage("floodFillMask",& IplImage(temp));)*/
	//__debug(waitKey(0);)/**/

}


