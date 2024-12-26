#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

std::string GetFolderPath(const std::string& file_path)
{
	int index = file_path.find_last_of('/');
	if (index>-1)
	{
		return file_path.substr(0, index);
	}
    return file_path.substr(0, file_path.find_last_of('\\'));
}

int main(){
    //  "/root/work/hello-world.jpg" ---->  "/root/work"
    string file_path = "/root/work/hello-world.jpg";
    string folder_path = GetFolderPath(file_path);
    printf("%s\n", folder_path.c_str());
    return 0;
}