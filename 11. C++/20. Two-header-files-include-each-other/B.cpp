#include <stdio.h>
#include "B.h"              // 注意, 这里既要包含B的头文件, 也要包含A的头文件(且B在前)
#include "A.h"


void B::printb(){
    printf("Value b is %d\n", b);
}

void B::printa(A* objA){
    objA->printa();
}
