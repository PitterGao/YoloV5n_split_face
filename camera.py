import time
import cv2
import os
import onnx
import numpy as np
import onnxruntime as ort
from utils.tools import filter_box, draw


class Camera:
    """
    camera_origin: the source of the frame, such as : webcam, mp4
    """

    def __init__(self, camera_origin=0):
        self.frame_buffer = []
        self.origin = camera_origin
        self.video = cv2.VideoCapture(self.origin)
        print(f'[Camera]: Camera Inited!')

    def init_video(self):
        print('[Camera]: Video Heating!')
        count = 0
        while count < 60:
            ret, frame = self.video.read()
            if not ret:
                break
            count += 1
        print('[Camera]: Video Inited!')

    def get_frames(self, f_gap, bs):
        """
        :param f_gap:
        :param bs:
        @details press 'q' to quit the camera in show mode
        """
        count = 0
        frame_num = 0
        self.frame_buffer = []

        if not self.video.isOpened():
            print(f"[Camera]: Error Opening Video Capture!")
            return

        # Read frames
        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                break
            count += 1
            if count % f_gap == 0:
                frame_num += 1
                frame = cv2.resize(frame, (640, 640))
                frame = frame[:, :, ::-1].transpose((2, 0, 1))
                frame = frame.astype(dtype=np.float32) / 255.0
                self.frame_buffer.append(frame)
                if len(self.frame_buffer) >= bs:
                    break

        return self.frame_buffer

    def save_frames(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)

        for i, frame in enumerate(self.frame_buffer):
            frame_output_path = os.path.join(folder_path, f"frame_{i + 1}.jpg")
            cv2.imwrite(frame_output_path, frame)

        print(f"[Camera]: Frames Saved Successfully!")

    def show_video(self):
        for frame in self.frame_buffer:
            cv2.imshow("Video", frame)
            if cv2.waitKey(0) & 0xFF == ord('n'):
                break

        cv2.destroyAllWindows()

    def close(self):
        self.video.release()
        print('[Camera]: Camera Closed!')


if __name__ == '__main__':
    camera = Camera()
    camera.init_video()
    camera.get_frames(1, 2)
    camera.show_video()
