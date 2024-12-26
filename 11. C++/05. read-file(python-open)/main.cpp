#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

vector<unsigned char> load_file(const string& file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
};

int main(){
    auto data = load_file("test.txt");
    for (auto& c : data)
        cout << c;
    cout << endl;
    return 0;
};