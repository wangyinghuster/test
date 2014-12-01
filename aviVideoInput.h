#ifndef AVIVIDEOINPUT_H
#define AVIVIDEOINPUT_H

#include "videoInput.h"

using namespace cv;
using namespace std;

class aviVideoInput  : public videoMOGInput
{
public:
    aviVideoInput(string filePath);

private:
    string aviPath;

};

#endif // AVIVIDEOINPUT_H
