import os
import glob
import cv2
import numpy as np
from PIL import Image


def get_obs_image(cfg):
    cap = cv2.VideoCapture(0)
    def func():
        ret, frame = cap.read()  # get a frame
        if cfg.show:
            cv2.imshow("frame", frame)
        if cfg.rgb:
            return [Image.fromarray(frame).convert('RGB')]
        else:
            return [Image.fromarray(frame).convert('L')]
    return func
    
    
def get_folder_image(cfg):
    def func(directory_path: str):
        image_pattern = os.path.join(directory_path, "*.*g")
        image_files = glob.glob(image_pattern)
        if cfg.rgb:
            return [Image.open(i).convert('RGB') for i in image_files]
        else:
            return [Image.open(i).convert('L') for i in image_files]
        # TODO DM5 Hash & detect
    return func

if __name__ == "__main__":
    ...
    