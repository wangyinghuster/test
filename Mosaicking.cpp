#include "../include/Mosaicking.h"
//本文件是接口的实现，里面包含了数据转换函数以及定义的接口，所以这里编程的时候用了C和C++的混合编程。

#include "stdlib.h"
#include "stdio.h"
#include <iostream>
#include <fstream>
#include "stitchingProcess.h"

using namespace std;

#define VCOM_PREPARE_FRAME 20  //背景检测的帧数

//#define LOGFILE   //定义是否将debugInfo输出到log.txt中
#ifdef LOGFILE
ofstream logFile("Log.txt");
#endif

typedef struct vComhandle{   //handle的结构体
    stitchingProcess *vPro;
    long int mosakingNum;
    int channels;
	int stateFlag;//0表示正常，非0值表示不正常
} vComH;

//静态全局变量static防止外部引用
static vector<vComH> vComHandleSet; //handle集

//下面这两个函数是转换函数
int matToFrame(Mat &imgM,TagFrame &imgT){

    int rows=imgM.rows;
    int cols=imgM.cols;

    cols=cols-cols%4;  //生成4字节对齐的数据

    imgT.width=cols;
    imgT.height=rows;
    imgT.type=0; //始终生成RBG类型的数据
    imgT.len=imgT.width*imgT.height*3;  //分配数据长度

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

    if (imgT.type==1){  //如果类型是YUV那么转换为RGB //这里转换暂时有问题
        imgM.create(imgT.height,imgT.width,CV_8UC3);
        memcpy(imgM.data,imgT.data,imgT.len*sizeof(char));
        cvtColor(imgM,imgM,CV_YCrCb2RGB);
    }
    else { //对于RGB格式的文件这么搞
        imgM.create(imgT.height,imgT.width,CV_8UC3);  //使用create创造的矩阵永远是连续的

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
    cout.rdbuf(logFile.rdbuf()); //设置cout重定向

#endif

    sysException::install();   //修改异常的方法

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
    return -1;               //返回一个负值，表示处理因为运行时错误出现不可扭转的失败。。。所以少年，放弃吧。。。
}
catch(cv::Exception &e){   //抓取OpenCV的异常
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

int Mosaicking(long handle,TMosaickingInput *input,TFrame *output) try  //整个函数都挂上异常处理
{
	
	if (vComHandleSet[handle].stateFlag){  //这边是拼接出现了异常之后的处理
		//直接返回第一幅图的结果


		output->height=input->frame[0].height;  //返回参数
		output->width=input->frame[0].width; //4字节对齐
		output->len=(output->width)*(output->height)*3;
		//memcpy(output->data,input->frame[0].data,output->len);
		return -1;
	}

    long int &num=vComHandleSet[handle].mosakingNum;
    stitchingProcess &combInst= *(vComHandleSet[handle].vPro);

    if (input->nChannel!=vComHandleSet[handle].channels) throw sysException("Input channels not match!");
    int channels=vComHandleSet[handle].channels;
    TFrame *framIn=input->frame;

    num++;//访问次数加一

/*//目前这里没有什么要做的。
    if (num==1){  //第一次访问，初始化各个参数
        for (unsigned int i=0;i<videoSet.size();i++){
            videoSeq &videoI=videoSet[i];
          //  videoI.setResizePara(405,720);
        }
    }*/

    if (num>VCOM_PREPARE_FRAME)
	{  //进入拼接模式
        num--;   //这个值不再增加

        for (int i=0;i<channels;i++){
           //不太需要if (time!=img.timeStamp) return ERR_TIMESTAMP_NOT_MATCH; //时间戳不匹配的话

            TFrame &img=framIn[i];
            Mat imgM;
            frameToMat(img,imgM);       
            combInst.inputVideo(i,imgM);   //更新每一帧

        }

        Mat outputMat;
        combInst.stitch(outputMat); //生成拼接的前景
        if (outputMat.empty()) throw sysException("Output image empty,mosiacking faild!");
        matToFrame(outputMat,*output);  //转换并加上时间戳
        return __MOSAIKING_SUCCES_OUTPUT;
    }

    if (num<VCOM_PREPARE_FRAME)
	{  //前若干帧要准备
		//cout << channels <<","<< vComHandleSet[handle].channels << endl;
        for (int i=0;i<channels;i++){

            TFrame &img=framIn[i];
            Mat imgM;
            frameToMat(img,imgM);
            combInst.inputVideo(i,imgM);   //更新每一帧

        }

        return __MOSAIKING_PREPARING;
    }

    if (num==VCOM_PREPARE_FRAME) 
	{  //进行背景建模的运算

        for (int i=0;i<channels;i++){
            TFrame &img=framIn[i];
            Mat imgM;
            frameToMat(img,imgM);
            combInst.inputVideo(i,imgM);   //更新每一帧
        }

        Mat panoBGImg;
        combInst.prepare(panoBGImg);      //生成模型

        if (panoBGImg.empty()) throw sysException("Output image empty,mosiacking faild!");

        output->height=panoBGImg.rows;  //返回参数
        int cols=panoBGImg.cols;
        output->width=cols-cols%4;  //4字节对齐
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

	//出错之后的结尾工作
	vComHandleSet[handle].stateFlag=-1;
	output->height=input->frame[0].height;  //返回参数
	output->width=input->frame[0].width; //4字节对齐
	output->len=(output->width)*(output->height)*3;

    return -1;               //返回一个负值，表示处理因为运行时错误出现不可扭转的失败。。。所以少年，放弃吧。。。
}
catch(cv::Exception &e){   //抓取OpenCV的异常
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

	//出错之后的结尾工作
	vComHandleSet[handle].stateFlag=-1;
	output->height=input->frame[0].height;  //返回参数
	output->width=input->frame[0].width; //4字节对齐
	output->len=(output->width)*(output->height)*3;

	return -1;
}

int ReleaseMosaicking(long handle) try {  //释放这个空间
    delete (vComHandleSet[handle]).vPro;
    vComHandleSet[handle].vPro=NULL;
    return __MOSAIKING_SUCCES_OUTPUT;
}
catch(sysException &e){
    cout <<"[Runtime Error in Mosiacking]"<<e.what()<<endl;
    cout.flush();
    errorWindow eWin(e.what());
    eWin.showErr();
    return -1;               //返回一个负值，表示处理因为运行时错误出现不可扭转的失败。。。所以少年，放弃吧。。。
}
catch(cv::Exception &e){   //抓取OpenCV的异常
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
