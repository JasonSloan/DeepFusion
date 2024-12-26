#include <iostream>
#include <unordered_map>
#include <string>

int main() {
    std::unordered_map<std::string, int> myDict;

    myDict["apple"] = 5;
    myDict["banana"] = 10;
    myDict["orange"] = 7;

    // Access values by key
    std::cout << "Value of 'apple': " << myDict["apple"] << std::endl;

    // Check if a key exists
    if (myDict.find("banana") != myDict.end()) {
        std::cout << "Found 'banana'" << std::endl;
    }

    // Iterate over the dictionary
    std::cout << "Dictionary contents:" << std::endl;
    for (const auto& pair : myDict) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    return 0;
}
