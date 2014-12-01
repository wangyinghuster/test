#ifndef SYSEXCEPTION_H
#define SYSEXCEPTION_H

#include <Windows.h>
#undef max           //除去这三个宏定义，会与某些opencv的函数冲突
#undef min
#undef small

#include <string>
 
using namespace std;

class sysException : public exception
{
public:
    sysException(EXCEPTION_POINTERS const &) throw();
    sysException(struct _EXCEPTION_POINTERS *) throw();

    sysException(const char *msgChr) throw();
    sysException(const string &msgStr) throw();

    static void install() throw();
    const char* what() const throw();
private:
    string errorMessage;
}; 

class errorWindow
{
public:
	errorWindow(const char *errMsg) throw();
	errorWindow(const string &errMsg) throw();

    void showErr() throw();

private:
    string errorMsg;
};

#endif // SYSEXCEPTION_H
