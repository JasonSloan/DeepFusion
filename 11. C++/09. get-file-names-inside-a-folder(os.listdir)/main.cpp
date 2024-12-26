#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>     // opendir和readdir包含在这里
#include <cstring>      // strcmp包含在这里
#include <algorithm>    // std::sort包含在这里

using namespace std;

int main() {
    string input = "/opt/hdd/yolov5_cpp/images";
    vector<string> files_vector;
    // 打开文件夹
    DIR* pDir = opendir(input.c_str());

    if (!pDir) {
        cerr << "Error opening directory: " << strerror(errno) << endl;
        return 1;
    }

    struct dirent* ptr;
    // 读取文件夹中的文件
    while ((ptr = readdir(pDir)) != nullptr) {
        // strcmp比较两个字符串, 如果不是"."或者".."就继续
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            files_vector.push_back(input + "/" + ptr->d_name);
        }
    }

    // 关闭文件夹
    closedir(pDir);
    std::sort(files_vector.begin(), files_vector.end());

    // 打印文件夹下的文件名
    for (const string& file : files_vector) {
        cout << file << endl;
    }

    return 0;
}
