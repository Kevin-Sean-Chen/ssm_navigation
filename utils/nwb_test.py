# -*- coding: utf-8 -*-
"""
Created on Wed May 14 18:28:04 2025

@author: ksc75
"""

from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.file import Subject
from pynwb.misc import AnnotationSeries 

# %% example loading
test_nwb='C:/Users/ksc75/Downloads/2018_08_24_fly3_run1.nwb'
raw_io = NWBHDF5IO(test_nwb, "r")
nwb_in = raw_io.read()

# %% aiming for the raw image
nwb_in.processing["ophys"]["ImageSegmentation"]["ImagingPlane"]["voxel_mask"]


# %%
from pynwb import NWBHDF5IO

# Load NWB file
with NWBHDF5IO(test_nwb, 'r') as io:
    nwbfile = io.read()

    # Access the ophys module and segmentation
    ophys_module = nwbfile.processing['ophys']
    img_seg = ophys_module.data_interfaces['ImageSegmentation']
    plane_seg = img_seg.get_plane_segmentation()

    # How many ROIs?
    print(f"Number of ROIs: {len(plane_seg)}")

    # Get ROI masks (stored as a ragged array via VectorIndex + VectorData)
    # e.g., pixel masks
    pixel_mask = plane_seg['voxel_mask'][0]  # ROI 0 mask
    print("Pixel mask shape (ROI 0):", pixel_mask.shape)

    # Get other ROI fields
    roi_ids = plane_seg.id[:]
    print("ROI IDs:", roi_ids)

    # Access image masks if present (typically for full-frame masks)
    if 'voxel_mask' in plane_seg:
        mask = plane_seg['voxel_mask'][0]  # (H, W) binary image mask for ROI 0
        print("Image mask shape:", mask.shape)
