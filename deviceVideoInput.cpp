#include "deviceVideoInput.h"

deviceVideoInput::deviceVideoInput(int dNum)
{
    deviceNum=dNum;
    stringstream ss;
    ss <<"Device"<< dNum;
    ss >> name;
}
