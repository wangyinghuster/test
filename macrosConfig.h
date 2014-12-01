#ifndef MACROSCONFIG_H
#define MACROSCONFIG_H


/*
 *本文件是所有编译开关的一个集合，所以，所有的h、hpp文件都应该包含本文件
 *而且包含本文件的时候，请将include时候的本文件的位置置于所有的目录下的头文件之前。
 *确保本文件会最先被预处理器读取到
 */

//以下宏请在工程文件中定义
//#define  __MOSAIKING_API_DEF   //设置开发的dll导出设置，工程在编译的时候应该定义，外部程序引用mosaicking.h的时候不用定义

//#define  DEBUGINFO             //打开详细信息的输出
//#define  LOGFILE                //定义是否将debugInfo输出到log.txt中
//#define  MATRIXOUT             //输出标定程序的中间结果到Matrix.txt
//#define  IMGSHOW               //打开中间图像的输出
//#define  FINALOUT

//#define  OUTSHOW               //打开最终结果的图像输出
//*******************************//

//以下是编译开关宏

//1.库函数使用开关
//#define __NOT_USE_OCL_LIB       //Vc09 中没有ocl的库，所以在vs08编译的时候需要定义这个宏，这个宏可以专门配置

//2.OpenCV库函数使用开关
#define __USE_OPENCV_CLINYDR    //是否使用openCV的clinydr 的 warp 函数。定义了使用OpenCV的原始函数，否则使用自己写的
//#define __USE_OPENCV_MULBANDBLAND   //是否用opencv的multibandblander函数
//#define __USE_OPENCV_PYR          //multiBandBlend中是否使用自己写的简化版金字塔生成函数。定义了，使用OpenCV原始的库函数，否则用自己写的库

//3.拼接流程编译开关
#define __BLEND_METHOD  1             //定义融合方案，0：多波段融合，1：加权融合，2：Feature融合。其余的一概默认为加权融合
#define __DO_WAVE_CORRECT             //融合过程中是否使用波束校正
#define __EXPOSURE_COMPENSATOR      //融合过程中是否用曝光补偿 这个好像会内存不足的样子
//#define __SEAMFIND_GRAPHCUT_USE    //用何种拼接缝寻找方法，注释掉的时候则用Dp，否则是GraphCut，效果差不多

//4.拼接方式开关
//#define __USE_FOREGROUND_MASK         //是否启用前景拼接，开启了较慢...





//*******************************//

//以下是编译中的数值宏

//*******************************//


//以下宏逻辑不用更改！！！！！！

//1.融合方式的选择
#if __BLEND_METHOD==0
    #define __MULTIBAND_BLAND
#elif __BLEND_METHOD==1
    #define __WEIGHT_BLAND                 //目前呢，好像这个是最快的！！！！
#elif __BLEND_METHOD==2
    #define __FEATURE_BLAND
#else
    #define __WEIGHT_BLAND                 //默认使用加权融合，不然如果__BLEND_METHOD==3 居然会编译通过，只不过会跑崩而已了。。
#endif
//end//

//2.多波段融合函数的函数选择
#ifndef __MULTIBAND_BLAND
    #ifdef __USE_OPENCV_MULBANDBLAND
        #undef __USE_OPENCV_MULBANDBLAND
    #endif
#endif
//end//

//3.debug调试宏
#ifdef DEBUGINFO
#define __debug(...)  __VA_ARGS__
#else
#define __debug(...)
#endif


//**********************************//

#include "sysException.h"  //系统异常类定义文件，和系统有关

#endif // MACROSCONFIG_H
