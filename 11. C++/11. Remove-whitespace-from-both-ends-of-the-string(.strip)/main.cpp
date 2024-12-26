#include <stdio.h>
#include <string>

using namespace std;

void Trim(std::string& str){
    std::string::size_type pos = str.find_last_not_of(' ');
    if (pos != std::string::npos)
    {
        str.erase(pos + 1);
        pos = str.find_first_not_of(' ');
        if (pos != std::string::npos)
            str.erase(0, pos);
    }
    else
        str.erase(str.begin(), str.end());
}

int main(){
    // 去除字符串头尾的空格
    string str = "  hello world  ";
    Trim(str);
    printf("Trim: |%s|\n", str.c_str());
    return 0;
}