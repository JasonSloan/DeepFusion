#include <stdio.h>
#include <vector>
#include "infer.hpp"

using namespace std;

float mean(vector<float> x){
    float sum = 0;
    for (int i = 0; i < x.size(); ++i){
        sum += x[i];
    }
    return sum / x.size();
}

int main() {
    auto InferController = Infer();
    int niters = 100;
    for (int i = 0; i < niters; ++i){
        InferController.forward();
    }
    auto records = InferController.get_records();
    printf("preprocess time: %.2f, inference time: %.2f, postprocess time: %.2f\n", mean(records[0]), mean(records[1]), mean(records[2]));

}