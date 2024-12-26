#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

std::string ReplaceString(const std::string &str, const std::string &old_val, const std::string &new_val){
    std::string out = str;

    std::string::size_type old_val_size = old_val.size();
    std::string::size_type new_val_size = new_val.size();
    std::string::size_type pos = 0;
    while ((pos = out.find(old_val, pos)) != std::string::npos)
    {
        out.replace(pos, old_val_size, new_val);
        pos += new_val_size;
    }

    return out;
}

int main(){
    string str = "hello-world.jpg";
    string old_val = ".jpg";
    string new_val = ".png";
    string new_str  = ReplaceString(str, old_val, new_val);
    printf("%s\n", new_str.c_str());
    return 0;
}