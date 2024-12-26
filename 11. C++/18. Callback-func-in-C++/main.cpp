#include <stdio.h>
#include "myfunc.hpp"

// this "realcallback" function is defined by callers(other person)
void realcallback(float value){
    printf("The value is %f\n", value);
}

int main() {
    /* the "myfunc" function is defined by myself to realize sth
       for example, infer the model and set the result to callback function*/
    myfunc(&realcallback);
    return 0;
}