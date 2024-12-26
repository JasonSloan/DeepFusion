#include <string>
#include <iostream>
#include <stdarg.h>

using namespace std;

// type: a: append, w: write
bool writeToFile(const char* fileName, char type, const char szFormat[], ...)
{
    FILE *fp = nullptr;
    if (type == 'a' || type == 'A')
        fp = fopen(fileName, "a");
    else 
        fp = fopen(fileName, "w");

    if (!fp) {
        return false;
    }
    va_list ArgList;
    char szStr[1024 * 10];
    va_start(ArgList, szFormat);
    vsprintf(szStr, szFormat, ArgList);
    fprintf(fp, "%s", szStr);
    fclose(fp);

    return true;
}


int main() {
    char input[] = "Hello world!";
    string filename = "test.txt";
    writeToFile(filename.c_str(), 'w', "%s\n", input);
    // writeToFile(filename.c_str(), 'a', "%s\n", input);
    return 0;
}