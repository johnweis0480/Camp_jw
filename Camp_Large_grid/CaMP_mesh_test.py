import SimPEG.potential_fields as PF
from SimPEG import (
    utils, simulation, maps
)
import matplotlib.pyplot as plt
import numpy as np
from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz


# Reproducible Science
np.random.seed(518936)


#Create Survey
ccx = np.linspace(-8625, 8625, num=70)
ccy = np.linspace(-10625, 10625, num=86)

X, Y = np.meshgrid(
    ccx, ccy
)

#airborne magnetic survey
#altitude of 200 m
Z = 200. * np.ones_like(X)
rxLoc = np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T)]
print('number of data: ', rxLoc.shape[0])
rxLoc = PF.magnetics.receivers.Point(rxLoc, components=['tmi'])
inclination = 90.
declination = 0.
strength = 518936.0
inducing_field = (strength, inclination, declination)
srcField = PF.magnetics.sources.SourceField([rxLoc], parameters=inducing_field)
survey_mag = PF.magnetics.survey.Survey(srcField)


#Create Mesh
dhx,dhy,dhz = 250.,250.,100.  # minimum cell width (base mesh cell width)
nbcx = 128  # number of base mesh cells in x
nbcy = 128
nbcz = 64

# Define base mesh (domain and finest discretization)
hx = dhx*np.ones(nbcx)
hy = dhy*np.ones(nbcy)
hz = dhz*np.ones(nbcz)
mesh = TreeMesh([hx, hy, hz],x0='CCN')

# Define corner points for rectangular box
xp, yp, zp = np.meshgrid([-8750., 8750.], [-10750., 10750.],[-2000., 0.])
mesh_corners = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)] # mkvc creates vectors

# Discretize to finest cell size within rectangular box
mesh = refine_tree_xyz(mesh, mesh_corners, method="box", finalize=False)

mesh.finalize()  # Must finalize tree mesh before use




#Create model, add block of serpentinized
mtrue = np.zeros(mesh.nC)
mtrue = utils.model_builder.addBlock(mesh.gridCC, mtrue, np.r_[-2500, -5000, -1300], np.r_[2500, 5000, -300], 2)

#Create carbonated polygon, add to model
xp_c = np.array([-500, 500, -500, 500,-1000,1000,-1000,1000])
yp_c = np.kron([-5000.0, 5000.0], np.ones((4)))
zp_c = np.kron(np.ones((2)), [-1300, -1300, -300, -300])
xyz_c_pts = np.c_[mkvc(xp_c), mkvc(yp_c), mkvc(zp_c)]
ind_polygon = utils.model_builder.PolygonInd(mesh, xyz_c_pts)
mtrue[ind_polygon] = 1

'''
plt.gca().set_aspect('equal')
plt.gca().scatter(X,Y,s=10, label='data points grav and mag')
plt.gca().legend()

mm = mesh.plotSlice((mtrue),normal='Z', ind=-5, grid=True,pcolorOpts={'cmap':'jet','alpha':0.75},ax=plt.gca(),showIt = True)

mm = mesh.plotSlice((mtrue),normal='Y', grid=True,pcolorOpts={'cmap':'jet','alpha':0.75},ax=plt.gca(),showIt = True)
'''
#Magnetic model,foward simulation
mmag = mtrue.copy()
mmag[mtrue==1] = 0.05 #susc
mmag[mtrue==2] = 0.15

simulation_mag = PF.magnetics.simulation.Simulation3DIntegral(
    survey=survey_mag,
    mesh=mesh,
    chiMap=maps.IdentityMap(nP=mesh.nC),
    actInd=np.ones_like(mtrue,dtype='bool'),
    #store_sensitivities="forward_only",
)
noise_floor = 1.
relative_error = 0.
# Compute predicted data for some model
dpred_mag = simulation_mag.make_synthetic_data(mmag,relative_error=relative_error,noise_floor=noise_floor, f=None, add_noise=True)

#Write mag files,mesh
mesh.writeUBC('mesh_CaMP_jw.ubc')
mesh.write_model_UBC("CaMP_magnetic_synthetic_model_jw.ubc", mmag)
utils.io_utils.write_mag3d_ubc('magnetic_data_jw.obs', dpred_mag)

#Gravity Model, forward simulation
# Define Gravity survey
Z = 1. * np.ones_like(X)
rxLoc = np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T)]
print('number of data: ', rxLoc.shape[0])
rxLoc = PF.gravity.receivers.Point(rxLoc)
srcField = PF.gravity.sources.SourceField([rxLoc])
survey_gravity = PF.gravity.survey.Survey(srcField)

#Make_gravModel
mgrav = mtrue.copy()
mgrav[mtrue==1] = .1 #g/cc
mgrav[mtrue==2] = -.2


simulation_grav = PF.gravity.simulation.Simulation3DIntegral(
    survey=survey_gravity,
    mesh=mesh,
    rhoMap=maps.IdentityMap(nP=mesh.nC),
    actInd=np.ones_like(mtrue,dtype='bool'),
)
noise_floor = 0.05
relative_error = 0.

# Compute predicted data for some model
dpred_grav = simulation_grav.make_synthetic_data(mgrav,relative_error=relative_error,noise_floor=noise_floor, f=None, add_noise=True)
mesh.writeUBC('mesh_CaMP_jw.ubc')
mesh.write_model_UBC("CaMP_grav_synthetic_model_jw.ubc", mgrav)
utils.io_utils.write_grav3d_ubc('grav_data_jw.obs', dpred_grav)
