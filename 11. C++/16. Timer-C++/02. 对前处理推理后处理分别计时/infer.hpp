#include <vector>
#include <chrono>
#include <thread>

using namespace std;
using time_point = chrono::high_resolution_clock;

template <typename Rep, typename Period>
float micros_cast(const std::chrono::duration<Rep, Period>& d) {
    return static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(d).count()) / 1000.;
}


class Infer{
public:
    Infer()=default;
    ~Infer()=default;

    void preprocess(){
        this_thread::sleep_for(chrono::milliseconds(5));
    }

    void do_infer(){
        this_thread::sleep_for(chrono::milliseconds(10));
    }

    void postprocess(){
        this_thread::sleep_for(chrono::milliseconds(15));
    }

    void forward(){
        auto start = time_point::now();
        auto stop = time_point::now();

        start = time_point::now();
        preprocess();
        stop = time_point::now();
        Infer::records[0].push_back(micros_cast(stop - start));

        start = time_point::now();
        do_infer();
        stop = time_point::now();
        Infer::records[1].push_back(micros_cast(stop - start));

        start = time_point::now();
        postprocess();
        stop = time_point::now();
        Infer::records[2].push_back(micros_cast(stop - start));
    }

    vector<vector<float>> get_records(){
        return Infer::records;
    }

private:
    static vector<vector<float>> records;           // 静态成员变量声明
};

vector<vector<float>> Infer::records(3);            // 静态成员变量定义, 长度为3
