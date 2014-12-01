#include "featuresMatcher.h"
#include<map>


featuresMatcher::featuresMatcher()
    :matchThr(0.6)
{
}

featuresMatcher::~featuresMatcher(){} //�����������������ṩһ������

void featuresMatcher::buildMatch(vector<Mat> &imgSet, vector<imgFeatures> &fSet, vector<mLog> &matchInfoOut)
{

    if (imgSet.size()!=fSet.size()) throw sysException("Wrong size of imgSet and featureSet!");
    matchInfoOut.clear();

    Ptr<DescriptorMatcher> match= DescriptorMatcher::create("FlannBased");

    int imgNums = (int)imgSet.size();
    matchMap.create(imgNums,imgNums,CV_8UC1);
    matchMap.setTo((unsigned char)-1); //��ʼ��Ϊ-1,����setTo�����ܸ�ֵ��������Ҫ����ǿ��ת��

    __debug(cout << "[Info]Matching "<< imgNums << " videos..." << endl;)

    // ��i*j/2��ѡ����ƥ��

    for (int i=0;i<imgNums;i++)
    {
        Mat &mati=fSet[i].backGroundFeature;

        for (int j=i+1;j<imgNums;j++)
        {
            Mat &matj=fSet[j].backGroundFeature;
            vector<vector<DMatch> > matchPointitoj;
            vector<vector<DMatch> > matchPointjtoi;
            match->knnMatch(mati,matj,matchPointitoj,2); //�ҳ������ν�
            match->knnMatch(matj,mati,matchPointjtoi,2);

            int *indexItoJ=new int[(int)matchPointitoj.size()];
            int *indexJtoI=new int[(int)matchPointjtoi.size()];

            //��һ��ɸѡ��ɸѡ����Ϊ���ڽ��ĵ�ĵ��
            for (unsigned int k=0;k<matchPointitoj.size();k++){
                indexItoJ[k]=matchPointitoj[k][0].trainIdx;
            }
            for (unsigned int k=0;k<matchPointjtoi.size();k++){
                indexJtoI[k]=matchPointjtoi[k][0].trainIdx;
            }

            vector<DMatch> matchP;
            vector<DMatch> matchRev;  //���ϵ
            for (int k=0;k<(int)matchPointjtoi.size();k++)//ɸѡ
            {
                if (indexItoJ[indexJtoI[k]]==k){
                    //�ڶ���ɸѡ�������ν�֮����ҪС��ĳ���ض�ֵ������ΪmatchThr
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
            {  //ƥ������ʮ���ͼ���ƥ�伯��

               __debug(cout <<"Accepted"<< endl;)

               mLog matLog;
               matLog.queryInx=j;
               matLog.trainInx=i;

               matLog.matchPointIndex=matchP;
               matLog.matchPointIndexRev=matchRev;

               matchInfoOut.push_back(matLog);
               matchMap.at<char>(j,i)=matchMap.at<char>(i,j)=(char)(matchInfoOut.size()-1);  //�������bitmap

            }
            __debug(else cout <<"Rejected"<< endl;)

        }
    }


}

void featuresMatcher::findSeperatedMatchSets(Mat &setsFlagOut)
{
     setsFlag.create(1,matchMap.cols,CV_8UC1);
     setsFlag.setTo((unsigned char)-1); //��ʼ��Ϊ-1,����setTo�����ܸ�ֵ��������Ҫ����ǿ��ת��

     char *flag=setsFlag.ptr<char>(0);   //������-1����ΪʲôҪ��unsigned char��������
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
             if (perRow[j]!=-1)   //˵����ƥ���ϵ����
             {
                 if ((flag[j]!=-1)&&(flag[j]!=nowFlag))  //�ϲ�����������
                 {
                     char flagTemp=flag[j];
                     for (int k=0;k<matchMap.cols;++k){   //����������ϵı�־
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

     setsFlagOut=setsFlag.clone(); //����������
}

void featuresMatcher::findLargestSets(vector<Mat> &imgSetInOut, vector<imgFeatures> &fSetInOut, vector<mLog> &matchInfoInOut,vector<bool> &videoIdxOut)
{

    char *flag=setsFlag.ptr<char>(0);
    int *count=new int[setsFlag.cols];

	//--__debug(cout<<"cols:"<<setsFlag.cols<<endl;)
    for (int i=0;i<setsFlag.cols;++i)   //��ʼ��
    {
        count[i]=0;
    }
	//--__debug(cout<<"cols:"<<setsFlag.cols<<endl;)

    for (int i=0;i<setsFlag.cols;++i)   //ͳ�Ƴ��ִ���
    {
        if (flag[i]!=-1)  count[flag[i]]++;
    }

	//--__debug(cout<<"cols:"<<setsFlag.cols<<endl;)
    int maxCount=0;
    int maxFlag=-1;
    for (int i=0;i<setsFlag.cols;++i)  //��ȡ���ֵ
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
    idxMap.setTo((unsigned char)-1); //��ʼ��Ϊ-1,����setTo�����ܸ�ֵ��������Ҫ����ǿ��ת��
    char *iMap=idxMap.ptr<char>(0);

    vector<Mat> newImgSet(imgSetInOut);
    vector<imgFeatures> newFSet(fSetInOut);
    vector<mLog> newMatchInfo(matchInfoInOut);

    imgSetInOut.clear();
    fSetInOut.clear();
    matchInfoInOut.clear();

    int idx=0;
    for (int i=0;i<setsFlag.cols;++i)   //�ü�
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

    for (unsigned int i=0;i<newMatchInfo.size();i++)  //��������
    {
        mLog &miTemp=newMatchInfo[i];
        miTemp.queryInx=iMap[miTemp.queryInx];
        miTemp.trainInx=iMap[miTemp.trainInx];

        if ((miTemp.trainInx!=-1)&&(miTemp.queryInx!=-1))  //�ڼ�����
        {
            matchInfoInOut.push_back(miTemp);
        }
    }   //����ƥ���ϵ�ü�֮���費��Ҫ��matchMap�����������ϵҲ���ü�����

/*--	__debug(cout<<" imgSetInOut.size:"<< imgSetInOut.size()
		<<",fSetInOut.size:"<<fSetInOut.size()
		<<",matchInfoInOut.size:"<<matchInfoInOut.size()
		<<endl;)*/
}
