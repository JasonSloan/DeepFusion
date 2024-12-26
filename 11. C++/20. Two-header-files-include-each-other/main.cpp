#include "A.h"
#include "B.h"

int main(){
    auto a = A();
    auto b = B();
    a.printb(&b);
    b.printa(&a);
}