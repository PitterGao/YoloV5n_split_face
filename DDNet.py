import torch
import onnxruntime
import numpy as np
from utils.tools import read_images_from_folder
from yolov5n_face import YOLODetector


# 加载数据
device = torch.device('cuda')

folder_path = "data/images"
CLASSES = ['face', 'eye', 'mouth']
image_list = read_images_from_folder(folder_path)
images = np.stack(image_list, axis=0)

# 加载ONNX模型
onnx_yolo_path = "weights/best_batch_5.onnx"
onnx_DeiT_path = "./weights/DeiT_batch_5.onnx"
DeiT = onnxruntime.InferenceSession(onnx_DeiT_path)
yolo = YOLODetector(classes=CLASSES, o_device=device, onnx_path=onnx_yolo_path)

# 准备输入数据
face, eye, mouth = yolo.Segmentation_of_target_objects(images)
input_dict = {
    DeiT.get_inputs()[0].name: eye.cpu().numpy(),
    DeiT.get_inputs()[1].name: mouth.cpu().numpy()
}

# 运行推理
outputs = DeiT.run(None, input_dict)

# 获取输出结果
output_tensor = outputs[0]
output = torch.from_numpy(output_tensor)

print(output)
