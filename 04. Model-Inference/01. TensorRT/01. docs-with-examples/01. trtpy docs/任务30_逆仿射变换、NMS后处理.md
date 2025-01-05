拉取代码：

trtpy get-series tensorrt-intergrate 

cd tensorrt-intergrate 

trtpy change-proj 1.2



```C++
// 步骤：解码输出值（过滤低置信度的框，获取类别标签，逆仿射变换）-->NMS
// decode box：从不同尺度下的预测框还原到原输入图上(包括:预测框，类被概率，置信度）
vector<vector<float>> bboxes;      // 初始化变量bboxes:[[x1, y1, x2, y2, conf, label], [x1, y1, x2, y2, conf, label]...]
float confidence_threshold = 0.25; // 置信度
float nms_threshold = 0.5;         // iou阈值
for (int i = 0; i < output_numbox; ++i)
{
  float *ptr = output_data_host + i * output_numprob; // 每次偏移85
  float objness = ptr[4];                             // 获得置信度
  if (objness < confidence_threshold)
    continue;

  float *pclass = ptr + 5;                                        // 获得类别开始的地址
  int label = max_element(pclass, pclass + num_classes) - pclass; // 获得概率最大的类别
  float prob = pclass[label];                                     // 获得类别概率最大的概率值
  float confidence = prob * objness;                              // 计算后验概率
  if (confidence < confidence_threshold)
    continue;

  // 中心点、宽、高
  float cx = ptr[0];
  float cy = ptr[1];
  float width = ptr[2];
  float height = ptr[3];

  // 预测框
  float left = cx - width * 0.5;
  float top = cy - height * 0.5;
  float right = cx + width * 0.5;
  float bottom = cy + height * 0.5;

  // 对应图上的位置
  float image_base_left = d2i[0] * left + d2i[2];                                                                     // x1
  float image_base_right = d2i[0] * right + d2i[2];                                                                   // x2
  float image_base_top = d2i[0] * top + d2i[5];                                                                       // y1，这里实际应该是d2i[4] * top+d2i[5];
  float image_base_bottom = d2i[0] * bottom + d2i[5];                                                                 // y2，这里实际应该是d2i[4] * bottom+d2i[5];
  bboxes.push_back({image_base_left, image_base_top, image_base_right, image_base_bottom, (float)label, confidence}); // 放进bboxes中
}
printf("decoded bboxes.size = %d\n", bboxes.size());

// nms非极大抑制
// 通过比较索引为5(confidence)的值来将bboxes所有的框排序
std::sort(bboxes.begin(), bboxes.end(), [](vector<float> &a, vector<float> &b)
          { return a[5] > b[5]; });
std::vector<bool> remove_flags(bboxes.size()); // 设置一个vector，存储是否保留bbox的flags
std::vector<vector<float>> box_result;         // box_result用来接收经过nms后保留的框
box_result.reserve(bboxes.size());             // 给box_result保留至少bboxes.size()个存储数据的空间

auto iou = [](const vector<float> &a, const vector<float> &b)
{
  float cross_left = std::max(a[0], b[0]);
  float cross_top = std::max(a[1], b[1]);
  float cross_right = std::min(a[2], b[2]);
  float cross_bottom = std::min(a[3], b[3]);

  float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
  float union_area = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]) + std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]) - cross_area;
  if (cross_area == 0 || union_area == 0)
    return 0.0f;
  return cross_area / union_area;
};

for (int i = 0; i < bboxes.size(); ++i)
{
  if (remove_flags[i])
    continue; // 如果已经被标记为需要移除，则continue

  auto &ibox = bboxes[i];        // 获得第i个box
  box_result.emplace_back(ibox); // 将该box放入box_result中，emplace_back和push_back基本一样，区别在于emplace_back是inplace操作
  for (int j = i + 1; j < bboxes.size(); ++j)
  { // 遍历剩余框，与box_result中的框做iou
    if (remove_flags[j])
      continue; // 如果已经被标记为需要移除，则continue

    auto &jbox = bboxes[j]; // 获得第j个box
    if (ibox[4] == jbox[4])
    { // 如果是同一类别才会做iou
      // class matched
      if (iou(ibox, jbox) >= nms_threshold) // iou值大于阈值，将该框标记为需要remove
        remove_flags[j] = true;
    }
  }
}
printf("box_result.size = %d\n", box_result.size());
```







