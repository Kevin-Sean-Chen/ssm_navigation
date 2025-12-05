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

# %% Specifiy batch of exp folders
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
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2025-4-17',
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2025-4-21',
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2025-5-1']  ### jittered ribbon and OU

### testing for two-ribbon environments (for spatial or temporal jitter)
# exp_list = [r'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/opto_rig/perturb_ribbon/2025-5-12',
#             r'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/opto_rig/perturb_ribbon/2025-5-15',
#             r'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/opto_rig/perturb_ribbon/2025-5-19',
#             r'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/opto_rig/perturb_ribbon/2025-5-31',
#             r'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/opto_rig/perturb_ribbon/2025-6-5']

### EPG preliminaryt
# exp_list = [r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\periodic_ribbon\2025-6-27',
#             r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\periodic_ribbon\2025-6-30',
#             r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\periodic_ribbon\2025-7-3']
# exp_list = [r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\periodic_ribbon\2025-7-3']

### gap crossing and perturbation data
exp_list = [r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\gap_cross\2025-09-06\kevin', ### gap crossing data, 117
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\gap_cross\2025-09-11\kevin', ### Kir, TNT
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\gap_cross\2025-09-12\kevin', ### TNT
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\gap_cross\2025-09-19\kevin', ### Kir, TNT, 117
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\gap_cross\2025-09-23\kevin', ### Kir, TNT, 117
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\gap_cross\2025-09-25\kevin', ### Kir, TNT
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\gap_cross\2025-09-30\kevin', ### Kir
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\gap_cross\2025-10-02\kevin', ### TNT and Kir
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\gap_cross\2025-10-07\kevin', ### good 117
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\gap_cross\2025-10-09\kevin', ### good 117
            r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\gap_cross\2025-10-16\kevin', ### good 117
            ]

# %% define conditions
### old pipeline with perturbed ribbons ###
target_file = "exp_matrix.pklz"
exp_type = 'jitter0p05'   ### 183
# exp_type = 'jitter0p0_' ### 142
# exp_type = '_OU_' ### 176
# exp_type = 'gaussianribbon_vial'
forbidden_subs = []
###########################################

### perturbation and gap crossing data ###
target_file = "exp_matrix.joblib"
exp_type = 'decreasing gap 60s'
# exp_type = 'increasing gap 60s'
forbidden_subs = ['Kir', 'TNT', 'OCL']
# forbidden_subs = ['OCL']
##########################################

# %% finding in all subfolders
pkl_files = []
for ll in range(len(exp_list)):
    root_dir = exp_list[ll]
    subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    
    # Loop through each subfolder to search for the target file
    for subfolder in subfolders:
        for dirpath, dirnames, filenames in os.walk(subfolder):
            # if target_file in filenames and exp_type in dirpath:
            if target_file in filenames and exp_type in subfolder and not any(bad in subfolder for bad in forbidden_subs):
                full_path = os.path.join(dirpath, target_file)
                pkl_files.append(full_path)
                print(full_path)
                