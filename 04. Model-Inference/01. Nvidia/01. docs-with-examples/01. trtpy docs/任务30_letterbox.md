拉取代码：

trtpy get-series tensorrt-intergrate 

cd tensorrt-intergrate 

trtpy change-proj 1.2



代码片段：letterbox

```C++
// letter box
auto image = cv::imread("car.jpg");
// 通过双线性插值对图像进行resize（input_width和input_height是模型接受的尺寸大小）
float scale_x = input_width / (float)image.cols;
float scale_y = input_height / (float)image.rows;
// 短边按比例缩放，长边按比例缩放后再pad，所以scale取较小值
float scale = std::min(scale_x, scale_y);
float i2d[6], d2i[6];
// resize图像，源图像和目标图像几何中心的对齐
// 仿射变换矩阵公式，套用即可
i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * image.cols + input_width + scale  - 1) * 0.5;
i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * image.rows + input_height + scale - 1) * 0.5;

cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);  // image to dst(network), 2x3 matrix
cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);  // dst to image, 2x3 matrix
cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);  // 计算反仿射变换矩阵，后处理会用到
// 执行仿射变换
cv::Mat input_image(input_height, input_width, CV_8UC3);
cv::warpAffine(image, input_image, m2x3_i2d, input_image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));  // 对图像做平移缩放旋转变换,可逆
// 可保存查看是否正确
cv::imwrite("input-image.jpg", input_image);

// 下面这一段完成3个功能BGR-->RGB HWC-->CHW 归一化
int image_area = input_image.cols * input_image.rows;
unsigned char* pimage = input_image.data;
float* phost_b = input_data_host + image_area * 0;
float* phost_g = input_data_host + image_area * 1;
float* phost_r = input_data_host + image_area * 2;
for(int i = 0; i < image_area; ++i, pimage += 3){
  // 注意这里的顺序rgb调换了
  *phost_r++ = pimage[0] / 255.0f;
  *phost_g++ = pimage[1] / 255.0f;
  *phost_b++ = pimage[2] / 255.0f;
}
```







