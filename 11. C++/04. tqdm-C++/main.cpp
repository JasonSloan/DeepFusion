#include <string>
#include <stdio.h>
#include <chrono>
#include <thread>
#include <vector>

#include "tqdm.hpp"

using namespace std;


int main() {

    std::vector<int> A = {1, 2, 3, 4, 5, 6};
    for (int a : tq::tqdm(A)){
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    for (int a : tq::trange(10)){
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    return 0;
}