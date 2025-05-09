# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 18:06:53 2025

@author: ksc75
"""


import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma

import pickle
import gzip
import glob
import os

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

import numpy.ma as ma
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %% meta analysis
### to pull out all folders from a list of experiment dates
### find the subfolders that have the right name

# %% for perturbed data
# root_dir = 'C:/Users/ksc75/Yale University Dropbox/users/kiri_choi/data/ribbon_sleap/2024-9-17/'  ### for lots of ribbon data
# root_dir = 'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/opto_rig/odor_vision/2024-11-5'
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2024-11-7'  ### for full field and OU
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\100424_new'  ### OU ribbons
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2024-10-31' ### OU ribbons... need signal!

exp_list = [r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\100424_new',  ### old OU
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2025-3-20',  ### jittered ribbon
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2025-3-24',  ### jittered ribbon
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2025-3-31',  ### jittered ribbon
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2025-4-4',  ### jittered ribbon and OU
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2025-4-7',  ### jittered ribbon and OU
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2025-4-10',  ### jittered ribbon and OU
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2025-4-14',
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2025-4-17']  ### jittered ribbon and OU

target_file = "exp_matrix.pklz"
exp_type = 'jitter0p05'   ### 143
# exp_type = 'jitter0p0_' ### 92
# exp_type = '_OU_' ### 126

pkl_files = []
for ll in range(len(exp_list)):
    root_dir = exp_list[ll]
    subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    
    # Loop through each subfolder to search for the target file
    for subfolder in subfolders:
        for dirpath, dirnames, filenames in os.walk(subfolder):
            if target_file in filenames and exp_type in dirpath:
                full_path = os.path.join(dirpath, target_file)
                pkl_files.append(full_path)
                print(full_path)