#include <stdio.h>

#include "test.h"

using namespace std;


bool func_with_unfixed_params(std::string param0, ...){
    // 获得可变参数列表
    va_list arg;
    // 开始访问可变参数列表
    /*  传入离可变参数"..."最近的那个形参,
        在va_start宏内部，param0参数用于计算第一个可变参数的地址。
        这是通过将param0的地址减去一个固定值来实现的，
        这个固定值与param0参数的类型有关。*/
    va_start(arg, param0);  
    // 获取第一个参数        
    int param1 = va_arg(arg, int);
    // float类型的param2要传double类型
    // 获取第二个参数   
    float param2 = va_arg(arg, double);
    // 结束访问可变参数列表
    va_end(arg);
    printf("param2: %d, param3: %f\n", param1, param2);
    return true;
}
