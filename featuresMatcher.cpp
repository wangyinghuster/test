#include "featuresMatcher.h"
#include<map>


featuresMatcher::featuresMatcher()
    :matchThr(0.6)
{
}

featuresMatcher::~featuresMatcher(){} //纯虚析构函数必须提供一个定义

void featuresMatcher::buildMatch(vector<Mat> &imgSet, vector<imgFeatures> &fSet, vector<mLog> &matchInfoOut)
{

    if (imgSet.size()!=fSet.size()) throw sysException("Wrong size of imgSet and featureSet!");
    matchInfoOut.clear();

    Ptr<DescriptorMatcher> match= DescriptorMatcher::create("FlannBased");

    int imgNums = (int)imgSet.size();
    matchMap.create(imgNums,imgNums,CV_8UC1);
    matchMap.setTo((unsigned char)-1); //初始化为-1,但是setTo不接受负值，所以需要进行强制转换

    __debug(cout << "[Info]Matching "<< imgNums << " videos..." << endl;)

    // 对i*j/2对选择做匹配

    for (int i=0;i<imgNums;i++)
    {
        Mat &mati=fSet[i].backGroundFeature;

        for (int j=i+1;j<imgNums;j++)
        {
            Mat &matj=fSet[j].backGroundFeature;
            vector<vector<DMatch> > matchPointitoj;
            vector<vector<DMatch> > matchPointjtoi;
            match->knnMatch(mati,matj,matchPointitoj,2); //找出最近与次近
            match->knnMatch(matj,mati,matchPointjtoi,2);

            int *indexItoJ=new int[(int)matchPointitoj.size()];
            int *indexJtoI=new int[(int)matchPointjtoi.size()];

            //第一轮筛选，筛选出互为最邻近的点的点对
            for (unsigned int k=0;k<matchPointitoj.size();k++){
                indexItoJ[k]=matchPointitoj[k][0].trainIdx;
            }
            for (unsigned int k=0;k<matchPointjtoi.size();k++){
                indexJtoI[k]=matchPointjtoi[k][0].trainIdx;
            }

            vector<DMatch> matchP;
            vector<DMatch> matchRev;  //逆关系
            for (int k=0;k<(int)matchPointjtoi.size();k++)//筛选
            {
                if (indexItoJ[indexJtoI[k]]==k){
                    //第二轮筛选，最近与次近之比需要小于某个特定值，这里为matchThr
                    float dif1=(matchPointjtoi[k][0].distance)/(matchPointjtoi[k][1].distance);
                    float dif2=(matchPointitoj[indexJtoI[k]][0].distance)/(matchPointitoj[indexJtoI[k]][1].distance);

                    if ((dif1<matchThr)&&(dif2<matchThr)){
                        matchP.push_back(matchPointjtoi[k][0]);
                        matchRev.push_back(matchPointitoj[indexJtoI[k]][0]);
                    }
                }
            }
            delete [] indexItoJ;
            delete [] indexJtoI;

            int matchPoints=matchP.size();

            __debug(cout << "[Info]Finding "<< matchPoints <<" point pairs between img "<< i <<" and img "<<j<<"...";)

            if (matchPoints>10)
            {  //匹配点大于十个就加入匹配集合

               __debug(cout <<"Accepted"<< endl;)

               mLog matLog;
               matLog.queryInx=j;
               matLog.trainInx=i;

               matLog.matchPointIndex=matchP;
               matLog.matchPointIndexRev=matchRev;

               matchInfoOut.push_back(matLog);
               matchMap.at<char>(j,i)=matchMap.at<char>(i,j)=(char)(matchInfoOut.size()-1);  //保存这个bitmap

            }
            __debug(else cout <<"Rejected"<< endl;)

        }
    }


}

void featuresMatcher::findSeperatedMatchSets(Mat &setsFlagOut)
{
     setsFlag.create(1,matchMap.cols,CV_8UC1);
     setsFlag.setTo((unsigned char)-1); //初始化为-1,但是setTo不接受负值，所以需要进行强制转换

     char *flag=setsFlag.ptr<char>(0);   //明明有-1啊，为什么要用unsigned char！！！！
     char flagCount=0;
     for (int i=0;i<matchMap.rows;++i)
     {

         char *perRow=matchMap.ptr<char>(i);
         char nowFlag;
         nowFlag=flag[i];
         if (nowFlag==-1) {
             nowFlag=flagCount;
             ++flagCount;
             flag[i]=nowFlag;
         }

         for (int j=0;j<matchMap.cols;++j)
         {
			 //--__debug(cout <<(int)perRow[j]<<",";)
             if (perRow[j]!=-1)   //说明有匹配关系存在
             {
                 if ((flag[j]!=-1)&&(flag[j]!=nowFlag))  //合并这两个集合
                 {
                     char flagTemp=flag[j];
                     for (int k=0;k<matchMap.cols;++k){   //更新这个集合的标志
                         if (flag[k]==nowFlag) flag[k]=flagTemp;
                     }
                     nowFlag=flag[j];
                 }else{
                     flag[j]=nowFlag;
                 }

             }

         }
		 //--__debug(cout <<endl;)
     }

     setsFlagOut=setsFlag.clone(); //保存这个结果
}

void featuresMatcher::findLargestSets(vector<Mat> &imgSetInOut, vector<imgFeatures> &fSetInOut, vector<mLog> &matchInfoInOut,vector<bool> &videoIdxOut)
{

    char *flag=setsFlag.ptr<char>(0);
    int *count=new int[setsFlag.cols];

	//--__debug(cout<<"cols:"<<setsFlag.cols<<endl;)
    for (int i=0;i<setsFlag.cols;++i)   //初始化
    {
        count[i]=0;
    }
	//--__debug(cout<<"cols:"<<setsFlag.cols<<endl;)

    for (int i=0;i<setsFlag.cols;++i)   //统计出现次数
    {
        if (flag[i]!=-1)  count[flag[i]]++;
    }

	//--__debug(cout<<"cols:"<<setsFlag.cols<<endl;)
    int maxCount=0;
    int maxFlag=-1;
    for (int i=0;i<setsFlag.cols;++i)  //求取最大值
    {
        if (count[i]>maxCount)
        {
            maxCount=count[i];
            maxFlag=i;
			//--__debug(cout<<"maxCount:"<<maxCount<<endl;)
        }
    }
    delete [] count;

    idxMap.create(1,setsFlag.cols,CV_8UC1);
    idxMap.setTo((unsigned char)-1); //初始化为-1,但是setTo不接受负值，所以需要进行强制转换
    char *iMap=idxMap.ptr<char>(0);

    vector<Mat> newImgSet(imgSetInOut);
    vector<imgFeatures> newFSet(fSetInOut);
    vector<mLog> newMatchInfo(matchInfoInOut);

    imgSetInOut.clear();
    fSetInOut.clear();
    matchInfoInOut.clear();

    int idx=0;
    for (int i=0;i<setsFlag.cols;++i)   //裁剪
    {
        if (flag[i]==maxFlag)
        {
            iMap[i]=idx;
			videoIdxOut.push_back(true);
            ++idx;
            imgSetInOut.push_back(newImgSet[i]);
            fSetInOut.push_back(newFSet[i]);
		}else{
			videoIdxOut.push_back(false);
		}
    }

    for (unsigned int i=0;i<newMatchInfo.size();i++)  //调整索引
    {
        mLog &miTemp=newMatchInfo[i];
        miTemp.queryInx=iMap[miTemp.queryInx];
        miTemp.trainInx=iMap[miTemp.trainInx];

        if ((miTemp.trainInx!=-1)&&(miTemp.queryInx!=-1))  //在集合内
        {
            matchInfoInOut.push_back(miTemp);
        }
    }   //这里匹配关系裁剪之后，需不需要把matchMap里面的索引关系也给裁减掉？

/*--	__debug(cout<<" imgSetInOut.size:"<< imgSetInOut.size()
		<<",fSetInOut.size:"<<fSetInOut.size()
		<<",matchInfoInOut.size:"<<matchInfoInOut.size()
		<<endl;)*/
}
