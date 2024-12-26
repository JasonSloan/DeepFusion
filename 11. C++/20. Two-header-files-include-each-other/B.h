#pragma once
#include <stdio.h>
#include "A.h"

class A;                        // 注意, 虽然包含了A的头文件, 但是也要class A声明一下

class B{
public:
    void printb();
    void printa(A* objA);
private:
    int b{100};
};