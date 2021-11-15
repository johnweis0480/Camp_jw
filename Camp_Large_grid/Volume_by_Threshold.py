import discretize as ds
import SimPEG.potential_fields as pf
from SimPEG import (
    maps, utils, simulation, inverse_problem, inversion, optimization, regularization, data_misfit, directives, Data
)
from SimPEG.utils import io_utils
import numpy as np
import matplotlib.pyplot as plt

def volume_estimate(model,mesh,boundary,threshold):
    model_loc = np.ones_like(model)
    model_loc[np.where(mesh.cell_centers[:,0]<boundary[0])] = 0
    model_loc[np.where(mesh.cell_centers[:,0]> boundary[1])] = 0
    model_loc[np.where(mesh.cell_centers[:, 1] < boundary[2])] = 0
    model_loc[np.where(mesh.cell_centers[:, 1] > boundary[3])] = 0
    model_loc[np.where(mesh.cell_centers[:, 2] < boundary[4])] = 0
    model_loc[np.where(mesh.cell_centers[:, 2] > boundary[5])] = 0
    model_loc[np.where(model < threshold[0])] = 0
    model_loc[np.where(model > threshold[1])] = 0

    volume = np.sum(mesh.vol[np.where(model_loc==1)])

    return(volume)

mesh = ds.TreeMesh.read_UBC('mesh_CaMP_jw.ubc')
true_geology_g = mesh.read_model_UBC('CaMP_grav_synthetic_model_jw.ubc')
inversion_model_grav_L2 = mesh.read_model_UBC('CaMP_gravity_synthetic_inversion_model_jw.ubc')
inversion_model_grav = mesh.read_model_UBC('CaMP_gravity_synthetic_inversion_model_LPLQ_jw.ubc')
'''
inversion_model_mag = mesh.read_model_UBC('CaMP_magnetic_synthetic_inversion_model_jw.ubc')
inversion_model_LPLQ_mag = mesh.read_model_UBC('CaMP_magnetic_synthetic_inversion_model_LPLQ_jw.ubc')

threshes = np.linspace(.01,.15,15)
thresh_list = []
volume_list = []
for i in threshes:
    volume = volume_estimate(inversion_model_mag,mesh,boundary = [-5000,5000,-10000,10000,-2000,0],threshold=[i,1000])
    km3 = volume / (10 ** 9)
    thresh_list.append(i)
    volume_list.append(km3)



##
plt.plot(thresh_list,volume_list,label = 'L2 Inversion Model Serpentinized Volume Estimate')
plt.plot([.01,.15],[35,35],label = 'Synthetic Model Serpentinized Volume')
plt.xlabel('Mag. Sus. Threshold (SI)')
plt.ylabel('Volume Estimate (Km^3)')


threshes = np.linspace(.01,.15,15)
thresh_list = []
volume_list = []
for i in threshes:
    volume = volume_estimate(inversion_model_LPLQ_mag,mesh,boundary = [-5000,5000,-10000,10000,-2000,0],threshold=[i,1000])
    km3 = volume / (10 ** 9)
    thresh_list.append(i)
    volume_list.append(km3)

plt.plot(thresh_list,volume_list,label = 'LPLQ Inversion Model Serpentinized Volume Estimate')
plt.legend()
plt.show()
##
'''
threshes = np.linspace(-.25,-.003,10)
thresh_list = []
volume_list = []
for i in threshes:
    volume = volume_estimate(inversion_model_grav_L2,mesh,boundary = [-5000,5000,-10000,10000,-2000,0],threshold=[-10,i])
    km3 = volume / (10 ** 9)
    thresh_list.append(i)
    volume_list.append(km3)

end=True

##
plt.plot(thresh_list,volume_list,label = 'Inversion Model Serpentinized Volume Estimate')
plt.plot([0,-.25],[35,35],label = 'Synthetic Model Serpentinized Volume')
plt.xlabel('Density Threshold (g/cc)')
plt.ylabel('Volume Estimate (Km^3)')

threshes = np.linspace(-.25,-.003,10)
thresh_list = []
volume_list = []
for i in threshes:
    volume = volume_estimate(inversion_model_grav,mesh,boundary = [-5000,5000,-10000,10000,-2000,0],threshold=[-10,i])
    km3 = volume / (10 ** 9)
    thresh_list.append(i)
    volume_list.append(km3)

end=True

plt.plot(thresh_list,volume_list,label = 'LpLq Inversion Model Serpentinized Volume Estimate')
plt.legend()
plt.show()
##