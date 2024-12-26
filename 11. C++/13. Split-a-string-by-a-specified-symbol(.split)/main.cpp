#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

void SplitString(const std::string &str, std::vector<std::string>& val, const std::string &delim){
    val.clear();

    std::string::size_type pos1, pos2;
    pos2 = str.find(delim);
    pos1 = 0;
    while (std::string::npos != pos2)
    {
        val.push_back(str.substr(pos1, pos2 - pos1));

        pos1 = pos2 + delim.size();
        pos2 = str.find(delim, pos1);
    }
    if (pos1 != str.length())
        val.push_back(str.substr(pos1));
}

int main(){
    // 按照","分割字符串
    string str = "hello world";
    std::vector<string> val;
    std::string delim = ",";
    SplitString(str, val, delim);
    for (int i = 0; i < val.size(); i++) {
        printf("%s\n", val[i].c_str());
    }
    return 0;
}