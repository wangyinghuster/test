#ifndef DEVICEVIDEOINPUT_H
#define DEVICEVIDEOINPUT_H

#include "videoInput.h"

using namespace cv;
using namespace std;

class deviceVideoInput
{
public:
    deviceVideoInput(int dNum);

protected:
    int deviceNum;
    string name;

};

class deviceMOGVideoInput : public videoMOGInput,public deviceVideoInput
{
public:
    deviceMOGVideoInput(int dNum)
        :videoMOGInput(),deviceVideoInput(dNum)
    {videoName=name;}

};

class deviceAVGVideoInput : public videoAVGInput,public deviceVideoInput
{
public:
    deviceAVGVideoInput(int dNum)
        :videoAVGInput(),deviceVideoInput(dNum)
    {videoName=name;}
};


#endif // DEVICEVIDEOINPUT_H
