import cv2
import numpy as np


def images_numpy_to_tensor(img_path):
    # (640,640,3) -> (1, 3, 640, 640)
    img = cv2.imread(img_path)
    or_img = cv2.resize(img, (640, 640))
    img = or_img[:, :, ::-1].transpose((2, 0, 1))
    img = img.astype(dtype=np.float32)
    img /= 255.0
    img = np.expand_dims(img, axis=0)

    return img

def images_tensor_to_numpy(img):
    # (1, 3, 640, 640) -> (640,640,3)
    # img = img.cpu().numpy()
    img = img.squeeze()
    img = img.transpose((1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    or_img = (img * 255.0).astype(np.uint8)

    return or_img
