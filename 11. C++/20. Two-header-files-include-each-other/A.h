#pragma once
#include <stdio.h>
#include "B.h"

class B;                        // 注意, 虽然包含了B的头文件, 但是也要class B声明一下

class A{
public:
    void printa();
    void printb(B* objB);
private:
    int a{10};
};