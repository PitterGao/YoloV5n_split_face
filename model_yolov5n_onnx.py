import cv2
import onnx
import torch
import numpy as np
import onnxruntime as ort

from utils import images_numpy_to_tensor
from utils.tools import filter_box, draw


class Yolov5ONNX(object):
    def __init__(self, onnx_path):
        """检查onnx模型并初始化onnx"""
        onnx_model = onnx.load(onnx_path)
        try:
            onnx.checker.check_model(onnx_model)
        except Exception:
            print("Model incorrect")
        else:
            print("Model correct")

        options = ort.SessionOptions()
        options.enable_profiling = True
        self.onnx_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()

    def get_input_name(self):
        """获取输入节点名称"""
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)

        return input_name

    def get_output_name(self):
        """获取输出节点名称"""
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)

        return output_name

    def get_input_feed(self, image_numpy):
        """获取输入numpy"""
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = image_numpy

        return input_feed

    def inference(self, img):
        """
        onnx_session 推理
        """
        input_feed = self.get_input_feed(img)
        pred = self.onnx_session.run(None, input_feed)[0]

        return pred

    def inference_frame(self, frame):
        """
        1. 将视频帧进行resize
        2. 图像转BGR2RGB和HWC2CHW（因为yolov5的onnx模型输入为 RGB：1 × 3 × 640 × 640）
        3. 图像归一化
        4. 图像增加维度
        5. onnx_session推理
        """
        or_img = cv2.resize(frame, (640, 640))
        img = or_img[:, :, ::-1].transpose((2, 0, 1))
        img = img.astype(dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)

        input_feed = self.get_input_feed(img)
        pred = self.onnx_session.run(None, input_feed)[0]

        return pred, or_img


if __name__ == "__main__":
    device = torch.device('cuda')

    # 加载数据
    CLASSES = ['face', 'eye', 'mouth']
    image_path = "data/images/Far_person.jpg"
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))
    Input = images_numpy_to_tensor(image_path)

    # 加载模型
    onnx_model_path = "weights/yolov5n_face_batch_1.onnx"
    yolov5 = Yolov5ONNX(onnx_model_path)

    # 模型推理
    pres = yolov5.inference(Input)

    # 处理推理结果
    outbox = filter_box(pres)
    image = draw(image, outbox, CLASSES)

    # 显示原始图像和推理结果
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
