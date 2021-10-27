import discretize as ds
import SimPEG.potential_fields as pf
from SimPEG import (
    maps, utils, simulation, inverse_problem, inversion, optimization, regularization, data_misfit, directives, Data
)
from SimPEG.utils import io_utils
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(518936)

def plot_true_geology(indsliceplot,component):
    utils.plot2Ddata(
        mesh.gridCC[indsliceplot][:, [0, component]], true_geology_1[indsliceplot], nx=200, ny=200,
        contourOpts={'alpha': 0},
        clim=[0., 0.15],
        ax=ax,
        level=True,
        ncontour=3,
        levelOpts={'colors': 'k', 'linewidths': 2},
        method='nearest'
    )

# ## Load Mesh,true geology from synthetic
mesh = ds.TreeMesh.read_UBC('mesh_CaMP_jw.ubc')
true_geology_g = mesh.read_model_UBC('CaMP_grav_synthetic_model_jw.ubc')
true_geology = mesh.read_model_UBC('CaMP_magnetic_synthetic_model_jw.ubc')
data_mag = io_utils.read_mag3d_ubc('magnetic_data_jw.obs')
data_grav = io_utils.read_grav3d_ubc('grav_data_jw.obs')
inversion_model_mag = mesh.read_model_UBC('CaMP_magnetic_synthetic_inversion_model_jw.ubc')
inversion_model_grav = mesh.read_model_UBC('CaMP_gravity_synthetic_inversion_model_jw.ubc')

#Car
true_geology_1 = np.copy(true_geology)

true_geology[true_geology==0.05] = 1
true_geology[true_geology==.15] = 2
'''
true_geology_g[true_geology_g==.1] = 1
true_geology_g[true_geology_g==-.2] = 2
b=np.linalg.norm(true_geology-true_geology_g)
'''


indz = -5
indslicezplot = mesh.gridCC[:,2] == mesh.vectorCCz[indz]

indy = 64
indsliceyplot = mesh.gridCC[:,1] == mesh.vectorCCy[indy]


# Plot true geology model

ticksize, labelsize = 14, 16

#geology model
geo_plot = mesh.plotSlice(true_geology,normal='Y',  clim=[0.,2], pcolorOpts={'cmap':'inferno_r'}, grid=True)
geocb = plt.colorbar(geo_plot[0], ticks=[0,1,2])
geocb.set_ticklabels(['BCKGRD','Carbonated','Serpentinized'])
geocb.ax.tick_params(labelsize=ticksize)
plt.show()
geo_plot=mesh.plotSlice(true_geology ,normal='Z',  clim=[0.,2],ind=-10, pcolorOpts={'cmap':'inferno_r'}, grid=True)
geocb = plt.colorbar(geo_plot[0], ticks=[0,1,2])
geocb.set_ticklabels(['BCKGRD','Carbonated','Serpentinized'])
geocb.ax.tick_params(labelsize=ticksize)
plt.show()

# plot mag
clim = []
clim.append(np.min(inversion_model_mag))
clim.append(np.max(inversion_model_mag))
clim_data = []
clim_data.append(np.min(data_mag.dobs))
clim_data.append(np.max(data_mag.dobs))
fig, ax = plt.subplots(1,1,figsize=(15,10))
ax.set_aspect('equal')



mesh.plotSlice(np.ones(mesh.nC),normal='Z', ind=int(-10), grid=True,pcolorOpts={'cmap':'Greys'},ax=ax)
mm = utils.plot2Ddata(data_mag.survey.receiver_locations, data_mag.dobs,
                      ax=ax,level=True, clim = clim_data,
                     nx=20,ny=20, dataloc=True,ncontour=12, shade=True,
                      contourOpts={'cmap':'Reds', 'alpha':0.8}, 
                      levelOpts={'colors':'k','linewidths':0.5}
                      )
ax.set_aspect(1)
ax.set_title('Magnetic data values and locations,\nwith mesh and geology overlays', fontsize=16)
cb = plt.colorbar(mm[0],orientation='horizontal')
cb.set_label('nT', fontsize=18)

#overlay true geology model for comparison


plot_true_geology(indslicezplot,1)

plt.subplots_adjust(hspace=-0.25,wspace=0.1)
plt.show()

fig, ax = plt.subplots(1,1,figsize=(15,10))
ax.set_aspect('equal')
mm = mesh.plotSlice((inversion_model_mag),normal='Z', ind=-5, grid=False,pcolorOpts={'cmap':'Reds'},ax=ax,clim=clim)

plot_true_geology(indslicezplot,1)

cb = plt.colorbar(mm[0],orientation='vertical')
cb.set_label('Susceptibility', fontsize=18)
plt.show()


fig, ax = plt.subplots(1,1,figsize=(15,10))
cb = plt.colorbar(mm[0],orientation='horizontal',ax=ax)
cb.set_label('Susceptibility', fontsize=18)
mm = mesh.plotSlice((inversion_model_mag),normal='Y', grid=False,pcolorOpts={'cmap':'Reds'},ax=ax,clim=clim)
plot_true_geology(indsliceyplot,2)

plt.show()



#Plot Gravity
clim = []
clim.append(np.min(inversion_model_grav))
clim.append(np.max(inversion_model_grav))
clim_data = []
clim_data.append(np.min(data_grav.dobs))
clim_data.append(np.max(data_grav.dobs))
fig, ax = plt.subplots(1,1,figsize=(15,10))
ax.set_aspect('equal')

mesh.plotSlice(np.ones(mesh.nC),normal='Z', ind=int(-10), grid=True,pcolorOpts={'cmap':'Greys'},ax=ax)
mm = utils.plot2Ddata(data_grav.survey.receiver_locations, data_grav.dobs,
                      clim=clim_data,
                      ax=ax,level=True,
                     nx=20,ny=20, dataloc=True,ncontour=12, shade=True,
                      contourOpts={'cmap':'coolwarm', 'alpha':0.8},
                      levelOpts={'colors':'k','linewidths':0.5}
                      )
ax.set_aspect(1)
ax.set_title('Gravity data values and locations,\nwith mesh and geology overlays', fontsize=16)
cb = plt.colorbar(mm[0],orientation='horizontal',ax=ax)
cb.set_label('mGal', fontsize=18)


plot_true_geology(indslicezplot,1)

plt.subplots_adjust(hspace=-0.25,wspace=0.1)
plt.show()

fig, ax = plt.subplots(1,1,figsize=(15,10))
ax.set_aspect('equal')
mm = mesh.plotSlice((inversion_model_grav),normal='Z', ind=-5,clim=clim, grid=False,pcolorOpts={'cmap':'seismic'},ax=ax)

plot_true_geology(indslicezplot,1)

cb = plt.colorbar(mm[0],orientation='vertical')
cb.set_label('Density', fontsize=18)
plt.show()

fig, ax = plt.subplots(1,1,figsize=(15,10))
cb = plt.colorbar(mm[0],orientation='horizontal',ax=ax)
cb.set_label('Density', fontsize=18)
mm = mesh.plotSlice((inversion_model_grav),normal='Y', grid=False,pcolorOpts={'cmap':'seismic'},ax=ax)
plot_true_geology(indsliceyplot,2)

plt.show()

