import logging
from flask import Flask, request, jsonify
from model import Model
from logger import get_root_logger
from queue import Queue
import psutil
import GPUtil

import cv2
import time

# ==================================================================
# 运行该代码之前先初始化环境, 执行命令source `trtpy env-source --print`
# ==================================================================
app = Flask(__name__)

# 初始化模型
inferController = Model(engine_path="/root/SuperResolution/flask/engine.trtmodel")
# 初始化日志, 仅为该页代码负责;C++动态库sr.so产生的日志由NanoLog负责
logger = get_root_logger(logger_name='server status', log_level=logging.INFO, log_file="logs/server_status.log")
# 最大队列长度,超过该长度会直接返回
max_queue_size = 100
queue = Queue(max_queue_size)
# 用于存储用户取消请求的id, 看思维导图
CACELLED_BUFFER = []
# 用于存储正在处理的id, 看思维导图
QUEUE_BUFFER = []
# 错误代码
CODES = [ 
    # ===============================/api/commit部分===============================                                                                            
    {'code': 500, 'ok': False, 'msg': 'Queue is full'},                                     # 0  队列满了
    {'code': 501, 'ok': False, 'msg': 'BinaryImage file is None'},                          # 1  输入的二进制文件为空
    {'code': 502, 'ok': False, 'msg': 'BinaryImage read failed'},                           # 2  二进制文件读取失败
    {'code': 503, 'ok': False, 'msg': 'Image data is None or id is None'},                  # 3  二进制文件为空或者id为空
    {'code': 504, 'ok': False, 'msg': 'Id is not unique, already found in queue'},          # 4  id不唯一
    {'code': 505, 'ok': False, 'msg': 'Image data recovery failed'},                        # 5  图片恢复成ndarray失败
    {'code': 506, 'ok': False, 'msg': 'Input image is too large or too small, \
input size must be within [32*32, 720*720]'},                                               # 6  输入图片太大或者太小
    {'code': 507, 'ok': False, 'msg': 'Input image should have 3 channels'},                # 7  输入图片通道数不为3
    {'code': 508, 'ok': False, 'msg': 'Model inference failure'},                           # 8  模型推理失败
    {'code': 509, 'ok': False, 'msg': 'Unknow error, probably input file \
is not an image file'},                                                                     # 9  未知错误
    {'code': 510, 'ok': False, 'msg': 'Commit canceled by user'},                           # 10 用户取消请求   
    # ================================/api/cancel部分===============================
    {'code': 200, 'ok': True, 'msg': 'Commit canceled success'},                            # 11 用户取消请求成功
    {'code': 501, 'ok': False, 'msg': 'Commit canceled failure, \
this id doesnt exist'},                                                                     # 12 用户取消请求失败
    # ================================/api/progress部分=============================
    {'code': 200, 'ok': True, 'msg': 'Success', 
    'data': {'total': None, 'current': None, 
             'memoryUsage':None, 'gpuMemoryUsage': None, 'cpuUsage': None}},                # 13 轮询成功
    {'code': 501, 'ok': False, 'msg': 'This id doesnt exist',
    'data': {'total': None, 'current': None, 
             'memoryUsage':None, 'gpuMemoryUsage': None, 'cpuUsage': None}},                # 14 轮询部分失败(未查到该id)
    {'code': 502, 'ok': False, 'msg': 'Failed to obtain the CPU usage or memory usage',
    'data': {'total': None, 'current': None, 
             'memoryUsage':None, 'gpuMemoryUsage': None, 'cpuUsage': None}},                # 15 轮询部分失败(获取CPU或者内存使用率失败)
    {'code': 503, 'ok': False, 'msg': 'Totally failed!\
     Failed to obtain the CPU usage or memory usage and the id doesnt exist either'},       # 16 轮询全部失败(获取CPU或者内存使用率失败且未查到该id)
    # ================================/部分================================
    {'code': 0, 'msg': 'Success', 'is_alive': None}                                         # -1 查询实例是否存活
]


# 查询实例是否存活
@app.route('/')
def index():
    is_alive = inferController.inferInstance is not None
    CODES[-1]['is_alive'] = is_alive
    return jsonify(CODES[-1])


# 用户提交请求的路由
@app.route('/api/commit', methods=['POST'])
def commit():
    if queue.qsize() == max_queue_size:
        logger.error(CODES[0]['msg'])
        return jsonify(CODES[0])
    
    try:
        # 获取请求参数: 图片数据和图片id, 如果没有图片数据或者图片id, 则返回错误信息
        file = request.files.get('binaryImage')
        id = request.form.get('id')

        # 图片数据为空
        if file == '':
            logger.error(CODES[1]['msg'])
            return jsonify(CODES[1])
        # 图片读取失败
        try:
            binaryImage = file.read()
        except Exception as e:
            logger.error(f"{CODES[2]['msg']}, the detail is :{e}")
            return jsonify(CODES[2])
        # 图片读取为空或者ID为空
        if binaryImage is None or (id == ''):
            logger.error(CODES[3]['msg'])
            return jsonify(CODES[3])
        # 如果ID在队列中或者在QUEUE_BUFFER, 则返回错误信息
        for item in queue.queue:
            if id == item['id'] or id in QUEUE_BUFFER:
                logger.error(CODES[4]['msg'])
                return jsonify(CODES[4])
            

        # 逻辑代码处理
        # 入队         
        queue.put({'id': id, 'image': binaryImage, 'cancel': False})
        while queue.qsize() > 0:
            # 获得数据
            item = queue.get()
            binaryImage = item['image']
            id = item['id']
            cancel = item['cancel']
            QUEUE_BUFFER.append(id)
            # 是否用户取消请求的数据,这种情况CANCEL_BUFFER中是没有该id的
            if cancel:
                QUEUE_BUFFER.remove(id)
                logger.info(f"{CODES[10]['msg']}, id is {id}")
                return jsonify(CODES[10])
            # 执行推理
            result = inferController.commit(binaryImage, id)
            code = result.code
            id = result.id
            
            # time.sleep(10)
            # 在执行推理过程中用户取消请求
            if id in CACELLED_BUFFER:
                QUEUE_BUFFER.remove(id)
                CACELLED_BUFFER.remove(id)
                logger.info(f"{CODES[10]['msg']}, id is {id}")
                return jsonify(CODES[10])

            # 剩下不管执行成功还是失败都要出队,而且要从QUEUE_BUFFER中删除对应id
            # 如果执行失败, 则返回错误信息(错误代码505-508, 在CODES中序号为6-9)
            for i in range(6, 10):
                if code == CODES[i]['code']:
                    QUEUE_BUFFER.remove(id)
                    logger.error(CODES[i]['msg'])
                    return jsonify(CODES[i])
            
            # 执行成功, 获取图片数据编码程.jpg格式然后转二进制
            imageData = result.output_array
            imageData = cv2.imencode(".jpg", imageData)[1].tobytes()
            # with open("output_image.jpg", "wb") as f:
            #     f.write(imageData)
            # 注意,并不是json串,而是二进制数据
            QUEUE_BUFFER.remove(id)
            return imageData
        
    # 服务器异常
    except Exception as e:
        logger.error(f"{CODES[9]}, the detail is :{e}")
        return jsonify(CODES[9])
    

# 用户取消请求的路由
@app.route('/api/cancel', methods=['POST'])
def cancel():
    id = request.form.get('id')
    if id == '':
        logger.error(f"{CODES[3]['msg']}, id is {id}")
        return jsonify(CODES[3])
    # 先判断队列中是否有要取消的id,如果有直接标记cancel为True
    for item in queue.queue:
        if item['id'] == id:
            item['cancel'] = True
            logger.info(f"{CODES[11]['msg']}, id is {id}")
            return jsonify(CODES[11])
    # 如果没有,判断id是否在QUEUE_BUFFER中
    if id in QUEUE_BUFFER:
        CACELLED_BUFFER.append(id)
        logger.info(f"{CODES[11]['msg']}, id is {id}")
        return jsonify(CODES[11])
    # 如果不在QUEUE_BUFFER中,则说明要取消的id已经被处理完毕或者不存在,直接返回错误信息   
    else:
        logger.error(f"{CODES[12]['msg']}, id is {id}")
        return jsonify(CODES[12])
                

# 轮询进度的路由
@app.route('/api/progress', methods=['POST'])
def progress():
    id = request.form.get('id')
    if id == '':
        logger.error(f"{CODES[3]['msg']}, id is {id}")
        return jsonify(CODES[3])
    try:
        init = {'total': None, 'current': None, 'memoryUsage':None, 'cpuUsage': None}
        # 获取队列长度
        total = queue.qsize()
        init['total'] = total + 1 if QUEUE_BUFFER else total
        # 判断队列中是否有要查询的id,如果有则记录队列中的位置
        for i, item in enumerate(queue.queue):
            if item['id'] == id:
                init['current'] = i + 1 if QUEUE_BUFFER else i
                break
        # 如果没有,则判断id是否在QUEUE_BUFFER中
        if init['current'] is None:
            if id in QUEUE_BUFFER:
                init['current'] = 0
        # 查询CPU, GPU和内存使用率
        cpu_usage = psutil.cpu_percent(interval=0.05)   # 50ms秒内的cpu使用率的平均值
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        gpu = GPUtil.getGPUs()[0]
        gpu_memory_usage = gpu.memoryUsed / gpu.memoryTotal * 100
        init['cpuUsage'] = f'{cpu_usage}%'
        init['memoryUsage'] = f'{memory_usage}%'
        init['gpuMemoryUsage'] = f'{gpu_memory_usage:.1f}%'
    except Exception as e:
        pass
    finally:
        if init['current'] is None and (init['cpuUsage'] is None or init['memoryUsage'] is None or init['gpuMemoryUsage'] is None):
            CODES[16]['data'] = init
            logger.error(f"{CODES[16]['msg']}, id is {id}")
            return jsonify(CODES[16])
        elif init['cpuUsage'] is None or init['memoryUsage'] is None or init['gpuMemoryUsage'] is None:
            CODES[15]['data'] = init
            logger.error(f"{CODES[15]['msg']}, id is {id}")
            return jsonify(CODES[15])
        elif init['current'] is None:
            print("======================================")
            CODES[14]['data'] = init
            logger.error(f"{CODES[14]['msg']}, id is {id}")
            return jsonify(CODES[14])
        else:
            CODES[13]['data'] = init
            return jsonify(CODES[13])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7000, debug=True)



