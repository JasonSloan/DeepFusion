#include <stdio.h>
#include "A.h"      // 注意, 这里既要包含A的头文件, 也要包含B的头文件(且A在前)
#include "B.h"


void A::printa(){
    printf("Value a is %d\n", a);
}

void A::printb(B* objB){
    objB->printb();
}