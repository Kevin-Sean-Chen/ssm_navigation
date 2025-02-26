# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:10:07 2025

@author: ksc75
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% load xlsx
# Replace 'your_file.xlsx' with the actual path to your Excel file
spec_file = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\spectrum_test\LED_spectrum_run.xlsx'  # G
spec_file = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\spectrum_test\LED_spectrum_proj.xlsx' # R
# spec_file = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\spectrum_test\LED_spectrum_screen_blue.xlsx' # B
# Read the Excel file
df = pd.read_excel(spec_file)

# %% extract specturm
data = df.iloc[54:-1].values.tolist()  # remove other text
specs = np.zeros((2, len(data)))

for ii in range(len(data)):
    input_string = data[ii][0]
    split_values = input_string.split(';')
    specs[:,ii] = np.array([float(split_values[0]), float(split_values[1])])

pos = np.argmax(specs[1,:])
max_nm = round(specs[0,pos],1)
# %% plotting
plt.figure()
plt.plot(specs[0,:], specs[1,:])
# plt.plot(spec_r[0,:], spec_r[1,:]/np.max(spec_r[1,:]),'r')
# plt.plot(spec_g[0,:], spec_g[1,:]/np.max(spec_g[1,:]),'g')
# plt.plot(spec_b[0,:], spec_b[1,:]/np.max(spec_b[1,:]),'b')
plt.xlabel('wavelength (nm)')
plt.ylabel('intensity (a. u.)')
plt.title('projector, peak at '+ str(max_nm)+' nm')
plt.xlim([600, 700])