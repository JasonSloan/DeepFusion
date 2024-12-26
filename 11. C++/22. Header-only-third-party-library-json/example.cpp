#include <string>
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

int main(){
    json j;
    float n1 = 80;
    float n2 = 80;
    j["n1"] = n1;
    j["n2"] = n2;
    // 序列化和反序列化
    string j_str = j.dump();            
    float n1 = j.at("n1").get<float>();             
    float n2 = j.at("n2").get<float>();
}