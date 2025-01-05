
#include <cuda_runtime.h>

static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy){
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void decode_kernel(
    float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
    float* invert_affine_matrix, float* parray, int max_objects, int NUM_BOX_ELEMENT
){  
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;
    // gpu中每一个线程处理一个box的值（85个），所以获取这85个值的首地址就是predict的地址偏移（线程索引*85）
    float* pitem     = predict + (5 + num_classes) * position;
    // x, y, w, h, objectness, class_confidence
    float objectness = pitem[4]; 
    // 如果objectness小于阈值，就不处理这个box   
    if(objectness < confidence_threshold)
        return;
    // 获取类别1的地址
    float* class_confidence = pitem + 5;
    // 获取类别1的置信度赋值给confidence，并使class_confidence指向类别2的地址
    float confidence        = *class_confidence++;
    int label               = 0;
    // 遍历类别2到类别80，找到最大的类别置信度，赋值给confidence，并记录类别索引
    // i等于几，就是类别几+1
    for(int i = 1; i < num_classes; ++i, ++class_confidence){
        if(*class_confidence > confidence){
            confidence = *class_confidence;
            label      = i;
        }
    }
    // 将类别置信度与objectness相乘，得到最终的置信度
    confidence *= objectness;
    // 如果最终的置信度小于阈值，就不处理这个box
    if(confidence < confidence_threshold)
        return;

    // 现成通信的函数，每个线程会在parray的第一个元素上加1，返回原来的值
    // 所以这也就是为什么parray的大小要设置成max_objects+1，因为第一个元素是用来记录box数量的
    int index = atomicAdd(parray, 1);
    if(index >= max_objects)
        return;

    float cx         = *pitem++;
    float cy         = *pitem++;
    float width      = *pitem++;
    float height     = *pitem++;
    float left   = cx - width * 0.5f;
    float top    = cy - height * 0.5f;
    float right  = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    // affine_project(invert_affine_matrix, left,  top,    &left,  &top);
    // affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    // left, top, right, bottom, confidence, class, keepflag
    // parray是output_device的地址，parray+1+index*NUM_BOX_ELEMENT是output_device的第index个元素的地址（但是这里为什么要+1）
    // 将满足要求的边框保存在output_device中
    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore
}

static __device__ float box_iou(
    float aleft, float atop, float aright, float abottom, 
    float bleft, float btop, float bright, float bbottom
){

    float cleft 	= max(aleft, bleft);
    float ctop 		= max(atop, btop);
    float cright 	= min(aright, bright);
    float cbottom 	= min(abottom, bbottom);
    
    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if(c_area == 0.0f)
        return 0.0f;
    
    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void fast_nms_kernel(float* bboxes, int max_objects, float threshold, int NUM_BOX_ELEMENT){

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    // TODO:？？为什么这里的bboxes用int*强转后就能计算box数量了？？？
    int count = min((int)*bboxes, max_objects);
    if (position >= count) 
        return;
    
    // left, top, right, bottom, confidence, class, keepflag
    // 当前线程处理的box的地址
    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    // 遍历所有的box
    for(int i = 0; i < count; ++i){
        // 获得第i个box的地址
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        // 如果第i个box和当前线程处理的box是同一个，或者第i个box的类别和当前线程处理的box的类别不同，就不处理这个box
        if(i == position || pcurrent[5] != pitem[5]) continue;

        // 只找置信度比当前线程处理的box的置信度大的box比如叫ibox，
        // 如果该ibox与当前线程处理的box的iou大于阈值，就把当前线程处理的box的keepflag置为0
        if(pitem[4] >= pcurrent[4]){
            // 如果恰巧第i个box和当前线程处理的box的置信度相等，且i小于当前线程处理的box的位置（说明在前面处理过了），就不处理这个box
            if(pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0],    pitem[1],    pitem[2],    pitem[3]
            );
            // 也就是站在当前线程所在的框，看看有没有比自己更牛逼的，有，那么当前线程所在的框就被抑制了
            if(iou > threshold){
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
} 

void decode_kernel_invoker(
    float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
    float nms_threshold, float* invert_affine_matrix, float* parray, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream){
    // num_bboxes:25200, num_classes:80, confidence_threshold:0.25, nms_threshold:0.45, max_objects:1000
    auto block = num_bboxes > 512 ? 512 : num_bboxes;   // 512
    auto grid = (num_bboxes + block - 1) / block;       // 51

    /* 如果核函数有波浪线，没关系，他是正常的，你只是看不顺眼罢了 */
    // invert_affine_matrix:nullptr, parray:output_device
    decode_kernel<<<grid, block, 0, stream>>>(
        predict, num_bboxes, num_classes, confidence_threshold, 
        invert_affine_matrix, parray, max_objects, NUM_BOX_ELEMENT
    );
    // block:512
    block = max_objects > 512 ? 512 : max_objects;
    // grid:3
    grid = (max_objects + block - 1) / block;
    fast_nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold, NUM_BOX_ELEMENT);
}