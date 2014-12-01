#ifndef MACROSCONFIG_H
#define MACROSCONFIG_H


/*
 *���ļ������б��뿪�ص�һ�����ϣ����ԣ����е�h��hpp�ļ���Ӧ�ð������ļ�
 *���Ұ������ļ���ʱ���뽫includeʱ��ı��ļ���λ���������е�Ŀ¼�µ�ͷ�ļ�֮ǰ��
 *ȷ�����ļ������ȱ�Ԥ��������ȡ��
 */

//���º����ڹ����ļ��ж���
//#define  __MOSAIKING_API_DEF   //���ÿ�����dll�������ã������ڱ����ʱ��Ӧ�ö��壬�ⲿ��������mosaicking.h��ʱ���ö���

//#define  DEBUGINFO             //����ϸ��Ϣ�����
//#define  LOGFILE                //�����Ƿ�debugInfo�����log.txt��
//#define  MATRIXOUT             //����궨������м�����Matrix.txt
//#define  IMGSHOW               //���м�ͼ������
//#define  FINALOUT

//#define  OUTSHOW               //�����ս����ͼ�����
//*******************************//

//�����Ǳ��뿪�غ�

//1.�⺯��ʹ�ÿ���
//#define __NOT_USE_OCL_LIB       //Vc09 ��û��ocl�Ŀ⣬������vs08�����ʱ����Ҫ��������꣬��������ר������

//2.OpenCV�⺯��ʹ�ÿ���
#define __USE_OPENCV_CLINYDR    //�Ƿ�ʹ��openCV��clinydr �� warp ������������ʹ��OpenCV��ԭʼ����������ʹ���Լ�д��
//#define __USE_OPENCV_MULBANDBLAND   //�Ƿ���opencv��multibandblander����
//#define __USE_OPENCV_PYR          //multiBandBlend���Ƿ�ʹ���Լ�д�ļ򻯰���������ɺ����������ˣ�ʹ��OpenCVԭʼ�Ŀ⺯�����������Լ�д�Ŀ�

//3.ƴ�����̱��뿪��
#define __BLEND_METHOD  1             //�����ںϷ�����0���ನ���ںϣ�1����Ȩ�ںϣ�2��Feature�ںϡ������һ��Ĭ��Ϊ��Ȩ�ں�
#define __DO_WAVE_CORRECT             //�ںϹ������Ƿ�ʹ�ò���У��
#define __EXPOSURE_COMPENSATOR      //�ںϹ������Ƿ����عⲹ�� ���������ڴ治�������
//#define __SEAMFIND_GRAPHCUT_USE    //�ú���ƴ�ӷ�Ѱ�ҷ�����ע�͵���ʱ������Dp��������GraphCut��Ч�����

//4.ƴ�ӷ�ʽ����
//#define __USE_FOREGROUND_MASK         //�Ƿ�����ǰ��ƴ�ӣ������˽���...





//*******************************//

//�����Ǳ����е���ֵ��

//*******************************//


//���º��߼����ø��ģ�����������

//1.�ںϷ�ʽ��ѡ��
#if __BLEND_METHOD==0
    #define __MULTIBAND_BLAND
#elif __BLEND_METHOD==1
    #define __WEIGHT_BLAND                 //Ŀǰ�أ�������������ģ�������
#elif __BLEND_METHOD==2
    #define __FEATURE_BLAND
#else
    #define __WEIGHT_BLAND                 //Ĭ��ʹ�ü�Ȩ�ںϣ���Ȼ���__BLEND_METHOD==3 ��Ȼ�����ͨ����ֻ�������ܱ������ˡ���
#endif
//end//

//2.�ನ���ںϺ����ĺ���ѡ��
#ifndef __MULTIBAND_BLAND
    #ifdef __USE_OPENCV_MULBANDBLAND
        #undef __USE_OPENCV_MULBANDBLAND
    #endif
#endif
//end//

//3.debug���Ժ�
#ifdef DEBUGINFO
#define __debug(...)  __VA_ARGS__
#else
#define __debug(...)
#endif


//**********************************//

#include "sysException.h"  //ϵͳ�쳣�ඨ���ļ�����ϵͳ�й�

#endif // MACROSCONFIG_H
