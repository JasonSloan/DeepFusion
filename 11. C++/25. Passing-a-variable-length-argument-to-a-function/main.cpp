#include <stdio.h>
#include <string>
#include "test.h"

using namespace std;

int main(){
    string param1 = "param1";
    int param2 = 2;
    float param3 = 3.0;
    func_with_unfixed_params(param1, param2, param3);
}