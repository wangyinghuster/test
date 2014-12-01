/*
���ļ��и�����������ƽ̨��صģ���windowsƽ̨�й�
*/

#include "sysException.h"
#include "eh.h"
#include <new.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <sstream>

static void sysExcep2cExcep(unsigned , EXCEPTION_POINTERS *info)  //����ϵͳ��SEH�쳣�������ʱ����Ҫ��  /EHa ����
{
   cout <<"[System Error Happened!!!]"<<endl;
   throw sysException(*info);                 
}

static void TerminateHandler() 
{ 
    cout << "[Runtime Error Happend!!!]Terminate handler called!" <<endl;
}

static void UnexpectedHandler() 
{ 
    cout << "[Runtime Error Happend!!!]Unexpected handler called!" <<endl;
}

LONG WINAPI unHandledExcep(struct _EXCEPTION_POINTERS *pExp)    //���ϲ��δ�����쳣����õĺ���,OpenCV�ڴ治���ʱ��ᱻ�����������
{  
	char tmpAddr[11];
	char tmpCode[11];

	stringstream sbuf;

	sprintf_s(tmpCode, "0x%X\0", pExp->ExceptionRecord->ExceptionCode);
	sbuf << "Exception code:" << tmpCode;

	sprintf_s(tmpAddr, "0x%X\0", *(unsigned long *)(pExp->ExceptionRecord->ExceptionAddress));
	sbuf << " at " << tmpAddr;

	cout <<"[Unhandled Error Happened!!!]"<<sbuf.str()<<endl;
	throw  sysException(sbuf.str());   //�ٶ��׳��쳣��C++���쳣���ƻ���
}  

int memoryAllocatedFailed( size_t memSize)                //�����ڴ�ʧ�ܵ�ʱ�����õĺ���,throw RuntimeError
{
    stringstream sbuf;
	sbuf <<"Memory allocated failed when allocate "<<memSize<<" bytes!" << endl;

	cout <<"[Memory Allocate Error Happened!!!]"<<sbuf.str()<<endl;
	throw sysException(sbuf.str());
	//return EXCEPTION_EXECUTE_HANDLER;
}

 void invalidParameterHandler(const wchar_t* expression,const wchar_t* errorFunction,const wchar_t* file,unsigned int line,uintptr_t pReserved)    //���÷Ƿ�������ʱ��ͻ�������������Ȼ������������׳�һ���쳣��
 {  
     //wprintf(L"Invalid parameter detected in function %s."L" File: %s Line: %d\n", function, file, line);
     //wprintf(L"Expression: %s\n", expression);

	 char str[100];
	 string msg="Invalid parameters detected in function:";

	 WideCharToMultiByte(CP_ACP, NULL,errorFunction,-1, str, 99, 0, 0);
	 if (strlen(str)==0) 
		 msg.append("(Unknown)");
	 else 
	     msg.append(str);

     msg.append(" in file:");

	 WideCharToMultiByte(CP_ACP, NULL,file,-1, str, 99, 0, 0);
	 if (strlen(str)==0) 
		 msg.append("(Unknown)");
	 else 
		 msg.append(str);

	 msg.append(" in line:");

	 stringstream sbuf;
	 sbuf << line;
	 msg.append(sbuf.str());

	 msg.append(" in expression:");

	 WideCharToMultiByte(CP_ACP, NULL,expression,-1, str, 99, 0, 0);
	 if (strlen(str)==0) 
		 msg.append("(Unknown)");
	 else 
		 msg.append(str);
	
	 cout <<"[Invalid parameters Error Happened!!!]"<<msg<<endl;
	 throw sysException(msg.c_str());
}

sysException::sysException(EXCEPTION_POINTERS const &info) throw()
{
    EXCEPTION_RECORD const &exception = *(info.ExceptionRecord);
    unsigned int code = exception.ExceptionCode;
    switch(code){
        case EXCEPTION_ACCESS_VIOLATION:
            errorMessage="Access violation!";           //Խ�����
            break;
        case EXCEPTION_INT_DIVIDE_BY_ZERO:
            errorMessage="Divide by zero!";             //�����
            break;
        case EXCEPTION_STACK_OVERFLOW:
            errorMessage="Stack overflow!";             //ջ���
            break;
        default:
            char num[20];
            sprintf_s(num, "0x%X\0",code);
            errorMessage="Default runtime error:"+string(num);
    }
}

sysException::sysException(struct _EXCEPTION_POINTERS *pExpInfo) throw()
{
	char tmpAddr[11];
	char tmpCode[11];

	stringstream sbuf;

	sprintf_s(tmpCode, "0x%X\0", pExpInfo->ExceptionRecord->ExceptionCode);
	sbuf << "Exception code:" << tmpCode;

	sprintf_s(tmpAddr, "0x%X\0", *(unsigned long *)(pExpInfo->ExceptionRecord->ExceptionAddress));
	sbuf << " at " << tmpAddr << endl;
    
	errorMessage.clear();
    sbuf >> errorMessage;

}

sysException::sysException(const char *msgChr) throw()
{
   errorMessage.clear();
   errorMessage.append(msgChr);
}

sysException::sysException(const string &msgStr) throw()
{
	errorMessage.clear();
	errorMessage.append(msgStr);
}

void sysException::install() throw()
{
   _set_se_translator(sysExcep2cExcep);                             //���ø��ֻص�����
   set_terminate(TerminateHandler);
   set_unexpected(UnexpectedHandler);
   SetUnhandledExceptionFilter(unHandledExcep);
   _set_new_handler(memoryAllocatedFailed);
   //_set_purecall_handler
   _set_invalid_parameter_handler(invalidParameterHandler);

}

const char* sysException::what() const throw()
{
    return errorMessage.c_str();
}

errorWindow::errorWindow(const char *errMsg) throw()
    :errorMsg(errMsg)
{
}

errorWindow::errorWindow(const string &errMsg) throw()
    :errorMsg(errMsg)
{
}

void errorWindow::showErr()
{
	errorMsg.append("\n\nExit this programe?");
    if(MessageBox(NULL, errorMsg.c_str(),TEXT("���ִ����ˣ���"),MB_ICONERROR|MB_YESNO)==IDYES)
		exit(-1);
}