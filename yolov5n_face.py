import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF
from yolov5.utils.general import non_max_suppression

from model_yolov5n_onnx import Yolov5ONNX
from utils.tools import read_images_from_folder, draw_detect


class YOLODetector(object):
    def __init__(self, onnx_path, classes, o_device, conf_thres=0.5, iou_thres=0.5):
        self.classes = classes
        self.device = o_device
        self.onnx_path = onnx_path
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.model = self.load_model()

    def load_model(self):
        model = Yolov5ONNX(self.onnx_path)
        return model

    def Segmentation_of_target_objects(self, batch_images, is_show=True):
        eye_tensor = torch.empty(0).to(self.device)
        face_tensor = torch.empty(0).to(self.device)
        mouth_tensor = torch.empty(0).to(self.device)

        pres = self.model.inference(batch_images)
        pres = torch.from_numpy(pres)
        outboxes = non_max_suppression(pres, self.conf_thres, self.iou_thres)
        bs, chl = batch_images.shape[0], batch_images.shape[1]

        for k, outbox in enumerate(outboxes):
            if is_show:
                img = draw_detect(batch_images[k], outbox, self.classes)
                cv2.imshow("检测效果图", img)
                cv2.waitKey(100)

            check_set = torch.unique(outbox[:, 5])
            if len(outbox) == 0 or len(check_set) != 3:
                if is_show:
                    draw_detect(batch_images[k], outbox, self.classes)
                    cv2.imshow("检测效果图", img)
                    cv2.waitKey(0)
                    print(f'[ERROR]Detect: Pred{len(pres)}, Object:{len(check_set)}')
                    exit(0)
                    # TODO: 提醒操作者脸部存在遮挡
                else:
                    print('[Detect] Jump Invalid Frames')
                    face_tensor = torch.zeros(size=(bs, chl, 224, 224), device=self.device)
                    eye_tensor = torch.zeros(size=(bs, chl, 224, 224), device=self.device)
                    mouth_tensor = torch.zeros(size=(bs, chl, 224, 224), device=self.device)
                    return face_tensor, eye_tensor, mouth_tensor

            eye_score = 0
            face_score = 0
            mouth_score = 0
            eye_image = torch.empty(0).to(self.device)
            face_image = torch.empty(0).to(self.device)
            mouth_image = torch.empty(0).to(self.device)
            batch_image_tensor = torch.from_numpy(batch_images).to(self.device)
            for i in range(len(outbox)):
                x1, y1, x2, y2, score, cls = outbox[i]
                x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), max(0, int(x2)), max(0, int(y2))
                if cls == 0:
                    if score > face_score:
                        face_score = score
                        face_image = batch_image_tensor[k, :, y1:y2, x1:x2]
                        face_image = TF.resize(face_image, [224, 224]).unsqueeze(0)
                elif cls == 1:
                    if score > eye_score:
                        eye_score = score
                        eye_image = batch_image_tensor[k, :, y1:y2, x1:x2]
                        eye_image = TF.resize(eye_image, [224, 224]).unsqueeze(0)
                else:
                    if score > mouth_score:
                        mouth_score = score
                        mouth_image = batch_image_tensor[k, :, y1:y2, x1:x2]
                        mouth_image = TF.resize(mouth_image, [224, 224]).unsqueeze(0)

            eye_tensor = torch.cat((eye_tensor, eye_image))
            face_tensor = torch.cat((face_tensor, face_image))
            mouth_tensor = torch.cat((mouth_tensor, mouth_image))

        return face_tensor, eye_tensor, mouth_tensor


if __name__ == '__main__':
    device = torch.device('cuda')

    folder_path = "data/images"
    CLASSES = ['face', 'eye', 'mouth']
    image_list = read_images_from_folder(folder_path)
    images = np.stack(image_list, axis=0)

    onnx_model_path = "weights/best_batch_5.onnx"
    yolo = YOLODetector(classes=CLASSES, o_device=device, onnx_path=onnx_model_path)

    face, eye, mouth = yolo.Segmentation_of_target_objects(images)
    print(face.shape, eye.shape, mouth.shape)
    print(face.device, eye.device, mouth.device)

