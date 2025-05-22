# -*- coding: utf-8 -*-
"""
Created on Tue May 13 19:03:37 2025

@author: ksc75
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_avi_as_array(filepath):
    cap = cv2.VideoCapture(filepath)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale (optional)
        frames.append(gray)

    cap.release()
    video_array = np.stack(frames, axis=0)  # shape: [T, H, W]
    return video_array


# %% load avi file

avi_file = r'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/vidual_stim/2025-5-13/screen_projector_calibration 10mm 60s LED255 screen_blue4_0/frames.avi'
avi_file = r'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/vidual_stim/2025-5-13/screen_projector_calibration full_field_flash_5 _15s_proj_15s_screen/frames.avi'
avi_file = r'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/vidual_stim/screen_projector_calibration full_screen_flash_15/frames.avi'
avi_file = r'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/vidual_stim/screen_projector_calibration full_screen_flash_16/frames.avi'
v_array = load_avi_as_array(avi_file)
# video = iio.imread(avi_file)  # shape: (T, H, W) or (T, H, W, C)

# %% analyze tensor
screen = v_array[:,750:,300:]
proj = v_array[:, 700:1500, 100:500]

screen_i = np.sum(np.sum(screen,1),1)
proj_i = np.sum(np.sum(proj,1),1)

# %%
plt.figure()
plt.plot(screen_i/np.max(screen_i))
plt.plot(proj_i/ np.max(proj_i))
plt.ylabel('normed intensity'); plt.xlabel('frames')