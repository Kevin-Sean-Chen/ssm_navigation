# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:45:58 2025

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

# %% set parameters
frame_rate = 60 # Hz
stim_T = 10
off_T = 50
T = stim_T + off_T # in seconds
time = np.arange(0, T, 1/frame_rate)
lt = len(time)
data = np.zeros((128, 640, lt))

# %% make tensor
for tt in range(lt):
    

# %% make video
def create_frame(data, frame):
    fig, ax = plt.subplots()
    ax.imshow(data[:, :, frame])
    ax.set_title(f'Iteration: {frame}')
    plt.colorbar(ax.images[0], ax=ax)
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