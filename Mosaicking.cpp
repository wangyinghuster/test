#include "../include/Mosaicking.h"
//���ļ��ǽӿڵ�ʵ�֣��������������ת�������Լ�����Ľӿڣ����������̵�ʱ������C��C++�Ļ�ϱ�̡�

#include "stdlib.h"
#include "stdio.h"
#include <iostream>
#include <fstream>
#include "stitchingProcess.h"

using namespace std;

#define VCOM_PREPARE_FRAME 20  //��������֡��

//#define LOGFILE   //�����Ƿ�debugInfo�����log.txt��
#ifdef LOGFILE
ofstream logFile("Log.txt");
#endif

typedef struct vComhandle{   //handle�Ľṹ��
    stitchingProcess *vPro;
    long int mosakingNum;
    int channels;
	int stateFlag;//0��ʾ��������0ֵ��ʾ������
} vComH;

//��̬ȫ�ֱ���static��ֹ�ⲿ����
static vector<vComH> vComHandleSet; //handle��

//����������������ת������
int matToFrame(Mat &imgM,TagFrame &imgT){

    int rows=imgM.rows;
    int cols=imgM.cols;

    cols=cols-cols%4;  //����4�ֽڶ��������

    imgT.width=cols;
    imgT.height=rows;
    imgT.type=0; //ʼ������RBG���͵�����
    imgT.len=imgT.width*imgT.height*3;  //�������ݳ���

    try {
        for (int i=0;i<rows;i++){
            char *len=(char *)imgM.ptr<uchar>(i);
            memcpy((imgT.data+i*cols*3),len,cols*3*sizeof(char));
        }
    }
    catch(...){
        throw sysException("Memcpy error!");
    }

    return 1;
}

int frameToMat(TagFrame &imgT,Mat &imgM){

    if (imgT.height<=0||imgT.width<=0||(imgT.len!=(imgT.width*imgT.height*3)))
        throw sysException("Wrong tagFrame input!");

    if (imgT.type==1){  //���������YUV��ôת��ΪRGB //����ת����ʱ������
        imgM.create(imgT.height,imgT.width,CV_8UC3);
        memcpy(imgM.data,imgT.data,imgT.len*sizeof(char));
        cvtColor(imgM,imgM,CV_YCrCb2RGB);
    }
    else { //����RGB��ʽ���ļ���ô��
        imgM.create(imgT.height,imgT.width,CV_8UC3);  //ʹ��create����ľ�����Զ��������

        try {
            memcpy(imgM.data,imgT.data,imgT.len*sizeof(char));
        }
        catch(sysException &e){
            throw sysException((string("TFrame Memcpy error:")+string(e.what())).c_str());
        }


    }

    return 1;
}

int InitMosaicking(long *handle,TInitPara para) try
{
#ifdef LOGFILE
    cout.rdbuf(logFile.rdbuf()); //����cout�ض���

#endif

    sysException::install();   //�޸��쳣�ķ���

    vComH vComb;
    vComb.mosakingNum=0;
	vComb.stateFlag=0;
    vComb.vPro=new stitchingProcess();
    vComb.channels=para.nChannel;

    vComb.vPro->generateVideoInput(vComb.channels);

    vComHandleSet.push_back(vComb);
    (*handle)=(long)vComHandleSet.size()-1;

    return __MOSAIKING_SUCCES_OUTPUT;
}
catch(sysException &e){
    cout <<"[Runtime Error in Mosiacking]"<<e.what()<<endl;
    cout.flush();
    errorWindow eWin(e.what());
    eWin.showErr();
    return -1;               //����һ����ֵ����ʾ������Ϊ����ʱ������ֲ���Ťת��ʧ�ܡ������������꣬�����ɡ�����
}
catch(cv::Exception &e){   //ץȡOpenCV���쳣
    stringstream eBuf;

    eBuf<<"Error:"<<e.err
        <<" in function ";
    if (e.func.length()==0)
        eBuf<<"(Unknown)";
    else
        eBuf<<e.func;

    eBuf<<" in file "<<e.file
        <<" line "<<e.line
        <<",ErrorCode:"<< e.code;

    cout<<"[OpenCV Error in Mosiacking]"<<eBuf.str()<<endl;
    cout.flush();

    errorWindow eWin(eBuf.str());
    eWin.showErr();

    return -1;
}

int Mosaicking(long handle,TMosaickingInput *input,TFrame *output) try  //���������������쳣����
{
	
	if (vComHandleSet[handle].stateFlag){  //�����ƴ�ӳ������쳣֮��Ĵ���
		//ֱ�ӷ��ص�һ��ͼ�Ľ��


		output->height=input->frame[0].height;  //���ز���
		output->width=input->frame[0].width; //4�ֽڶ���
		output->len=(output->width)*(output->height)*3;
		//memcpy(output->data,input->frame[0].data,output->len);
		return -1;
	}

    long int &num=vComHandleSet[handle].mosakingNum;
    stitchingProcess &combInst= *(vComHandleSet[handle].vPro);

    if (input->nChannel!=vComHandleSet[handle].channels) throw sysException("Input channels not match!");
    int channels=vComHandleSet[handle].channels;
    TFrame *framIn=input->frame;

    num++;//���ʴ�����һ

/*//Ŀǰ����û��ʲôҪ���ġ�
    if (num==1){  //��һ�η��ʣ���ʼ����������
        for (unsigned int i=0;i<videoSet.size();i++){
            videoSeq &videoI=videoSet[i];
          //  videoI.setResizePara(405,720);
        }
    }*/

    if (num>VCOM_PREPARE_FRAME)
	{  //����ƴ��ģʽ
        num--;   //���ֵ��������

        for (int i=0;i<channels;i++){
           //��̫��Ҫif (time!=img.timeStamp) return ERR_TIMESTAMP_NOT_MATCH; //ʱ�����ƥ��Ļ�

            TFrame &img=framIn[i];
            Mat imgM;
            frameToMat(img,imgM);       
            combInst.inputVideo(i,imgM);   //����ÿһ֡

        }

        Mat outputMat;
        combInst.stitch(outputMat); //����ƴ�ӵ�ǰ��
        if (outputMat.empty()) throw sysException("Output image empty,mosiacking faild!");
        matToFrame(outputMat,*output);  //ת��������ʱ���
        return __MOSAIKING_SUCCES_OUTPUT;
    }

    if (num<VCOM_PREPARE_FRAME)
	{  //ǰ����֡Ҫ׼��
		//cout << channels <<","<< vComHandleSet[handle].channels << endl;
        for (int i=0;i<channels;i++){

            TFrame &img=framIn[i];
            Mat imgM;
            frameToMat(img,imgM);
            combInst.inputVideo(i,imgM);   //����ÿһ֡

        }

        return __MOSAIKING_PREPARING;
    }

    if (num==VCOM_PREPARE_FRAME) 
	{  //���б�����ģ������

        for (int i=0;i<channels;i++){
            TFrame &img=framIn[i];
            Mat imgM;
            frameToMat(img,imgM);
            combInst.inputVideo(i,imgM);   //����ÿһ֡
        }

        Mat panoBGImg;
        combInst.prepare(panoBGImg);      //����ģ��

        if (panoBGImg.empty()) throw sysException("Output image empty,mosiacking faild!");

        output->height=panoBGImg.rows;  //���ز���
        int cols=panoBGImg.cols;
        output->width=cols-cols%4;  //4�ֽڶ���
        output->len=(output->width)*(output->height)*3;

        return __MOSAIKING_READY;
    }

    return __MOSAIKING_SUCCES_OUTPUT;
}
catch(sysException &e){
    cout <<"[Runtime Error in Mosiacking]"<<e.what()<<endl;
    cout.flush();
    errorWindow eWin(e.what());
    eWin.showErr();

	//����֮��Ľ�β����
	vComHandleSet[handle].stateFlag=-1;
	output->height=input->frame[0].height;  //���ز���
	output->width=input->frame[0].width; //4�ֽڶ���
	output->len=(output->width)*(output->height)*3;

    return -1;               //����һ����ֵ����ʾ������Ϊ����ʱ������ֲ���Ťת��ʧ�ܡ������������꣬�����ɡ�����
}
catch(cv::Exception &e){   //ץȡOpenCV���쳣
	stringstream eBuf;

	eBuf<<"Error:"<<e.err
		<<" in function ";
	if (e.func.length()==0)
		eBuf<<"(Unknown)";
	else
	    eBuf<<e.func;
	
	eBuf<<" in file "<<e.file 
		<<" line "<<e.line
		<<",ErrorCode:"<< e.code;

	cout<<"[OpenCV Error in Mosiacking]"<<eBuf.str()<<endl;
	cout.flush();

    errorWindow eWin(eBuf.str());
    eWin.showErr();

	//����֮��Ľ�β����
	vComHandleSet[handle].stateFlag=-1;
	output->height=input->frame[0].height;  //���ز���
	output->width=input->frame[0].width; //4�ֽڶ���
	output->len=(output->width)*(output->height)*3;

	return -1;
}

int ReleaseMosaicking(long handle) try {  //�ͷ�����ռ�
    delete (vComHandleSet[handle]).vPro;
    vComHandleSet[handle].vPro=NULL;
    return __MOSAIKING_SUCCES_OUTPUT;
}
catch(sysException &e){
    cout <<"[Runtime Error in Mosiacking]"<<e.what()<<endl;
    cout.flush();
    errorWindow eWin(e.what());
    eWin.showErr();
    return -1;               //����һ����ֵ����ʾ������Ϊ����ʱ������ֲ���Ťת��ʧ�ܡ������������꣬�����ɡ�����
}
catch(cv::Exception &e){   //ץȡOpenCV���쳣
    stringstream eBuf;

    eBuf<<"Error:"<<e.err
        <<" in function ";
    if (e.func.length()==0)
        eBuf<<"(Unknown)";
    else
        eBuf<<e.func;

    eBuf<<" in file "<<e.file
        <<" line "<<e.line
        <<",ErrorCode:"<< e.code;

    cout<<"[OpenCV Error in Mosiacking]"<<eBuf.str()<<endl;
    cout.flush();

    errorWindow eWin(eBuf.str());
    eWin.showErr();

    return -1;
}
