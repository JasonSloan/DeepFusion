import logging
from flask import Flask, request, jsonify
from model import Model
from logger import get_root_logger


import numpy as np 
import cv2
import time

app = Flask(__name__)

inferController = Model(engine_path="/root/SuperResolution/flask/engine.trtmodel")
logger = get_root_logger(logger_name='server status', log_level=logging.INFO, log_file="logs/server_status.log")


@app.route('/')
def index():
    is_alive = inferController.inferInstance is not None
    return jsonify({
        'code': 0,
        'msg': 'Success',
        'is_alive': is_alive
    })


@app.route('/commit', methods=['POST'])
def commit():
    try:
        # 1. 获取参数
        file = request.files.get('binaryImage')
        if file is None:
            return jsonify({
                'code': -100,
                'msg': 'binaryImage file is None'
            })
        binaryImage = file.read()

        id = request.form.get('id')
        if binaryImage is None or id is None:
            return jsonify({
                'code': -100,
                'id': id,
                'msg': 'Image data is None or id is None'
            })

        # 2. 逻辑代码处理
        result = inferController.commit(binaryImage, id)

        code = result.code
        id = result.id
        

        if code != 0:
            return jsonify({
                'code': code,
                'id': id,
                'msg': 'Inference engine execute failed'
            })
        imageData = result.output_array.reshape(1060, 1892, 3)
        imageData = cv2.imencode(".jpg", imageData)[1].tobytes()


        with open("output_image.jpg", "wb") as f:
            f.write(imageData)
        
        return imageData

    except Exception as e:
        logger.error(f"Server down, the detail is :{e}")
        return jsonify({
            'code': -200,
            'msg': f'Server down, the detail is :{e}'
        })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7000, debug=True)



