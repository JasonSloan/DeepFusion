#include <stdio.h>
#include <string>

using namespace std;

bool starts_with(const std::string &str, const std::string &starting) {
    if (str.length() >= starting.length()) {
        return str.compare(0, starting.length(), starting) == 0;
    }
    return false;
}

bool ends_with(const std::string &str, const std::string &ending) {
    if (str.length() >= ending.length()) {
        return str.compare(str.length() - ending.length(), ending.length(), ending) == 0;
    }
    return false;
}

int main() {
    string str = "IamlearningC++";
    bool start = starts_with(str, "Iam");
    bool end = ends_with(str, "C++");

    printf("Starts with 'Iam': %d\n", start);
    printf("Ends with 'C++': %d\n", end);
    return 0;
}