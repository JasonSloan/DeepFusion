from rknn.api import RKNN


if __name__ == '__main__':
    model_path = 'weights/last.onnx'
    platform = 'rk3588'                                     # choose from [rk3566|rk3588|rk3562]                                 
    model_type = 'fp'                                       # choose from ['i8', 'fp']
    do_quant = True if model_type == 'i8' else False
    dataset_path = "../dataset/train.txt"                   # 如果是要做量化的话,需要指定量化后精度恢复的训练数据,该路径为一个txt文件,txt文件中存储着各个图像路径
    output_path = "weights/last.rknn"

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[
                    [255, 255, 255]], target_platform=platform)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant, dataset=dataset_path)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()
