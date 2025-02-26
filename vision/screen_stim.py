# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:45:58 2025

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image


# %% compute bar speed and time
loop_per_second = 1
frame_rate = 60
displace_per_update = int(np.floor(680 / (60*loop_per_second)))
x_size, y_size = 640, 120

ITI_time = 3 ### in seconds
ITI_steps = ITI_time* frame_rate
stim_steps = loop_per_second* frame_rate

T = loop_per_second + ITI_time # in seconds
time = np.arange(0, T, 1/frame_rate)
lt = len(time)
data = np.zeros((x_size, y_size, lt))

# %% initialize bar
bar_width = 50
pattern = np.zeros((x_size, y_size))
pattern[:bar_width,:] = 1
data[:,:,0] = pattern

# %% make tensor
for tt in range(lt):
    if tt <= stim_steps:
        data[:,:,tt] = pattern
        pattern = np.roll(pattern, shift=displace_per_update, axis=0)
    else:
        data[:,:,tt] = 0
    
# %% make video
def create_frame(data, frame):
    fig, ax = plt.subplots()
    ax.imshow(data[:, :, frame].T)
    # ax.set_title(f'Iteration: {frame}')
    # plt.colorbar(ax.images[0], ax=ax)
    plt.close(fig)  # Close the figure to avoid displaying it
    return fig

# Create and save frames as images
frames = []
for frame in range(data.shape[2]):
    fig = create_frame(data, frame)
    fig.canvas.draw()
    # Convert the canvas to an image
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(Image.fromarray(image))

# Save frames as a GIF
frames[0].save('visual_stim.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)

# Check if the GIF plays correctly
from IPython.display import display, Image as IPImage
display(IPImage('visual_stim.gif'))