#include <unistd.h>
#include <sys/stat.h>

using namespace std;

int main() {
    char path[] = "images";
    // access是确认该目录是否存在
    // S_IRWXU是创建的文件夹的权限, 创建的为R(read)W(write)XU(exec)
    if (access(path, 0) != F_OK)
        mkdir(path, S_IRWXU);
    return 0;
};