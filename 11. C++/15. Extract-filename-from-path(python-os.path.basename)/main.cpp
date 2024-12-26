#include <string.h>     // 注意是string.h,不是string
#include <stdio.h>

using namespace std;

int main(){
    char fn[100]; 
    char fp[500] = "/root/image/bear.jpg";     
    char *ptr = strrchr(fp, '/'); 
    sprintf(fn, "%s", ptr + 1);      //加1为了去掉'/'
    printf("file name: %s\n", fn);
    return 0; 
}         
