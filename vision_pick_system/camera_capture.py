import numpy as np
from hik_camera import HikCamera


class Hik(HikCamera):
    def setting(self):
        self.setitem("ExposureAuto", "Continuous")
        try:
            self.set_rgb()      # 优先 RGB
        except AssertionError:
            self.set_raw(8)     # 不支持 RGB 则用 RAW8


class CameraCapture:
    """
    海康工业相机采集封装（基于 hik_camera SDK）
    输出统一为 BGR （兼容 OpenCV 后续处理）
    """

    def __init__(self, camera_ip: str):
        self.camera_ip = camera_ip
        self.cams = None
        self.cam = None

    def open(self):
        self.cams = Hik.get_cams([self.camera_ip])
        self.cams.__enter__()
        self.cam = self.cams[self.camera_ip]
        return self

    def read(self):
        if self.cam is None:
            raise RuntimeError("相机未打开，请先调用 open()")
        img = self.cam.get_frame()

        if self.cam.is_raw:
            img = self.cam.raw_to_uint8_rgb(img, poww=0.5)

        # SDK 输出按 RGB 处理，转为 OpenCV 常用 BGR
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img[..., ::-1]
        return np.ascontiguousarray(img)

    def capture_image(self, save_path=None):
        import cv2
        from pathlib import Path
        frame = self.read()
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), frame)
        return frame

    def close(self):
        if self.cams is not None:
            self.cams.__exit__(None, None, None)
            self.cams = None
            self.cam = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()