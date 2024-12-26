#include <string>
#include <iostream>
#include <stdarg.h>

using namespace std;

std::string GetFileName(const std::string& file_path, bool with_ext=true){
	int index = file_path.find_last_of('/');
	if (index < 0)
		index = file_path.find_last_of('\\');
    std::string tmp = file_path.substr(index + 1);
    if (with_ext)
        return tmp;

    std::string img_name = tmp.substr(0, tmp.find_last_of('.'));
    return img_name;
}


int main() {
    string file_path = "/root/work/imgs/dog.jpg";
    string img_name = GetFileName(file_path, true);
    printf("img name: %s\n", img_name.c_str());
    return 0;
}