import torch
import onnxruntime
import numpy as np
import time
from utils.tools import read_images_from_folder
from yolov5n_face import YOLODetector
from camera import Camera

# 加载相机
camera = Camera()
camera.init_video()

# 定义Meta数据
device = torch.device('cuda')
folder_path = "data/images"
CLASSES = ['face', 'eye', 'mouth']

# 加载ONNX模型
onnx_yolo_path = "weights/best_batch_5.onnx"
onnx_DeiT_path = "./weights/DeiT_batch_5.onnx"
DeiT = onnxruntime.InferenceSession(onnx_DeiT_path)
yolo = YOLODetector(classes=CLASSES, o_device=device, onnx_path=onnx_yolo_path)

while True:
    # 拿到Images
    images = np.stack(camera.get_frames(1, 5), axis=0)

    # 模型第一阶段
    face, eye, mouth = yolo.Segmentation_of_target_objects(images, is_show=False)
    input_dict = {
        DeiT.get_inputs()[0].name: eye.cpu().numpy(),
        DeiT.get_inputs()[1].name: mouth.cpu().numpy()
    }

    # 模型第二阶段
    outputs = DeiT.run(None, input_dict)

    # 获取输出结果
    output_tensor = outputs[0]
    output = torch.from_numpy(output_tensor)
    print(output)

