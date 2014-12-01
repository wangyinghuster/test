#include "frameBlender.h"

frameBlender::frameBlender()
    :prepared(false)
{
}

frameBlender::~frameBlender() {} //纯虚析构函数必须要提供一个定义

Point frameBlender::getAllSize(vector<Mat> &warpSeamMask,vector<Point> &topleft){
	//这里会将topleft的值给去负值化

    int xRight=0;
    int yBottom=0;
	int yTop=0;
    int xLeft=0;

    for (unsigned int i=0;i<topleft.size();i++){

		//--__debug(cout<<"tf.x:"<<topleft[i].x<<",tf.y:"<<topleft[i].y<<endl;)

        if (xRight<topleft[i].x+warpSeamMask[i].cols)
            xRight=topleft[i].x+warpSeamMask[i].cols;

        if (yBottom<topleft[i].y+warpSeamMask[i].rows)
            yBottom=topleft[i].y+warpSeamMask[i].rows;

		if (yTop>topleft[i].y)
			yTop=topleft[i].y;

		if (xLeft>topleft[i].x)
            xLeft=topleft[i].x;

    }

    outCols=xRight-xLeft;
    outRows=yBottom-yTop;
   
    //将topLeft定点给表示成正值
	for (unsigned int i=0;i<topleft.size();i++){

        topleft[i].y-=yTop;
        topleft[i].x-=xLeft;

        //--__debug(cout<<"tf.x:"<<topleft[i].x<<",tf.y:"<<topleft[i].y<<endl;)

	}

    Point rightBottom;
    rightBottom.x=outCols;
    rightBottom.y=outRows;

    return rightBottom;

}


