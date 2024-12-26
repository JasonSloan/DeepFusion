#include <string>
#include <iostream>
#include <sys/stat.h>

using namespace std;

int main() {
    string input = "images/person.jpg";
    struct stat info;
    if (stat(input.c_str(), &info) != 0) {
        cout << "Cannot find valid input file." << endl;
    }
    return 0;
}
