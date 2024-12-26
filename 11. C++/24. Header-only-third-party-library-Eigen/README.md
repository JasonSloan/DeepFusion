### 一. 编译安装

下载源码: [链接](https://github.com/JasonSloan/learning-notebook/files/15493309/eigen-3.3.9.zip)

编译安装:

```bash
unzip eigen-3.3.9.zip
cd eigen-3.3.9
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=eigen-build
sudo make install
```

### 二. 使用方法

eigen为纯头文件形式使用的库, 无需和源码一起再次编译, 只需指定include_directories即可

```C++
#include <iostream>
#include "eigen-build/include/eigen3/Eigen/Core"
#include "eigen-build/include/eigen3/Eigen/Dense"

using namespace std;

// 数据类型为int类型的行向量
typedef Eigen::Matrix<int, 1, 2> Vector2i;
// 数据类型为float类型的列向量
typedef Eigen::Matrix<float, 3, 1> Vector3f;
// 数据类型为double类型的列向量
typedef Eigen::Matrix<double, 4, 1> Vector4d;


int main() {
    // ! 注意, 如果为Eigen::Matrix类型, 那么行数和列数必须大于1
    // ! 打印的时候要使用cout不要使用printf, cout可以将矩阵或者向量一次性全部打印出来而不用遍历
    // 定义并初始化一个向量
    Vector3f w(1.0f , 2.0f , 3.0f);
    Vector3f v1 = Vector3f::Ones();
    Vector3f v2 = Vector3f::Zero();
    Vector4d v3 = Vector4d::Random();
    Vector4d v4 = Vector4d::Constant(1.8);
    // 加减操作
    cout << "向量与向量之间加减" << endl;
    cout << v1 + v2 << endl << endl;
    cout << v4 - v3 << endl << endl;
    // 向量乘标量
    cout << "向量乘标量" << endl;
    cout << v4 * 2 << endl << endl;
    // 向量与向量比较
    cout << "向量与向量比较" << endl;
    cout << (Vector3f::Ones() * 3 == Vector3f::Constant(3)) << endl << endl;
    // 向量和向量做矩阵乘法 3x1 * 1x3
    cout << "向量和向量做矩阵乘法 3x1 * 1x3" << endl;
    v1 = Vector3f::Random();
    v2 = Vector3f::Random();
    cout << v1 * v2.transpose() << endl << endl;
    // 矩阵和向量做矩阵乘法: 3x3 * 3x1
    cout << "矩阵和向量做矩阵乘法: 3x3 * 3x1" << endl;
    cout << Eigen::Matrix3f::Random() * v1 << endl << endl;
    // 向量和向量做点积
    cout << "向量和向量做点积" << endl;
    cout << v1.array() * v2.array() << endl << endl;


    // 一些特殊矩阵
    Eigen::Matrix3f A;
    Eigen::Matrix4d B;
    // [-1, 1]之间的均匀分布随机数矩阵
    cout << "[-1, 1]之间的均匀分布随机数矩阵" << endl;
    A = Eigen::Matrix3f::Random();
    cout << A << endl << endl;
    // 对角矩阵
    cout << "对角矩阵" << endl;
    B = Eigen::Matrix4d::Identity();
    cout << B << endl << endl;
    // 全0矩阵
    cout << "全0矩阵" << endl;
    A = Eigen::Matrix3f::Zero();
    cout << A << endl << endl;
    // 全1矩阵
    cout << "全1矩阵" << endl;
    A = Eigen::Matrix3f::Ones();
    cout << A << endl << endl;
    // 常数矩阵
    B = Eigen::Matrix4d::Constant(4.5);

    // 定义并初始化一个矩阵(矩阵元素类型为int, 2行3列, Eigen::RowMajor代表每一行内的元素在内存中存储是连续的)
    Eigen::Matrix<int, 2, 3, Eigen::RowMajor> matrix1;
    matrix1 << 1, 2, 3, 4, 5, 6;
    Eigen::Matrix<int, 3, 2, Eigen::RowMajor> matrix2;
    matrix2 << 1, 2, 3, 4, 5, 6;
    Eigen::Matrix4f M1 = Eigen::Matrix4f::Random();
    Eigen::Matrix4f M2 = Eigen::Matrix4f::Constant(2.2);

    // 矩阵与矩阵相加
    cout << "矩阵与矩阵相加" << endl;
    cout << M1 + M2 << endl ;
    // 矩阵转置
    cout << "矩阵转置" << endl;
    cout << M1.transpose () << endl ;
    // 矩阵求逆(不可逆的时候值为Nan) 
    cout << "矩阵求逆(不可逆的时候值为Nan) " << endl; 
    cout << M1 . inverse () << endl ;
    // 矩阵中每个元素做平方
    cout << "矩阵中每个元素做平方" << endl;
    cout << M1 . array () . square () << endl ;
    // 矩阵与矩阵做点积
    cout << "矩阵与矩阵做点积" << endl;
    cout << M1.array() * Eigen::Matrix4f::Identity().array() << endl ;
    // 矩阵和矩阵逐元素比较
    cout << "矩阵和矩阵逐元素比较" << endl;
    auto res = M1.array() <= M2.array();
    cout << res << endl;
    // 矩阵和矩阵做矩阵乘法: 2x3 * 3x2
    Eigen::Matrix<int, 2, 2> result = matrix1 * matrix2;
    cout << "矩阵和矩阵做矩阵乘法" << endl;
    cout << result << endl << endl;

    return 0;
}
```

