
import discretize as ds
import SimPEG.potential_fields as pf
from SimPEG import (
    maps, utils, simulation, inverse_problem, inversion, optimization, regularization, data_misfit, directives
)
from SimPEG.utils import io_utils
import numpy as np
import matplotlib.pyplot as plt

#Reproducible science
np.random.seed(518936)

mesh = ds.TreeMesh.read_UBC('mesh_CaMP_jw.ubc')

data_mag = io_utils.read_mag3d_ubc('magnetic_data_jw.obs')
data_grav = io_utils.read_grav3d_ubc('grav_data_jw.obs')

# ## Create simulations and data misfits
def run_grav_inversion():
    actvMap = maps.IdentityMap(mesh)

    # Grav problem
    simulation_grav = pf.gravity.simulation.Simulation3DIntegral(
        survey=data_grav.survey,
        mesh=mesh,
        rhoMap=actvMap,
    )
    dmis_grav = data_misfit.L2DataMisfit(data=data_grav, simulation=simulation_grav)

    # Initial Model
    m0 = np.zeros(mesh.n_cells)

    # # Sensitivity weighting

    # Define the regularization (model objective function).
    reg = regularization.Simple(mesh, mapping=actvMap, indActive=np.ones(mesh.n_cells, dtype=bool))

    # Define how the optimization problem is solved. Here we will use a projected
    # Gauss-Newton approach that employs the conjugate gradient solver.
    opt = optimization.ProjectedGNCG(
        maxIter=10, lower=-1.0, upper=1.0, maxIterLS=20, maxIterCG=10, tolCG=1e-3
    )

    # Here we define the inverse problem that is to be solved
    inv_prob = inverse_problem.BaseInvProblem(dmis_grav, reg, opt)

    # Defining a starting value for the trade-off parameter (beta) between the data
    # misfit and the regularization.
    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e0)
    beta_schedule = directives.BetaSchedule(coolingFactor=5, coolingRate=1)
    update_jacobi = directives.UpdatePreconditioner()
    target_misfit = directives.TargetMisfit(chifact=1)

    # Add sensitivity weights
    sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)

    # The directives are defined as a list.
    directives_list = [
        sensitivity_weights,
        starting_beta,
        beta_schedule,
        update_jacobi,
        target_misfit,
    ]

    inv3 = inversion.BaseInversion(inv_prob, directives_list)

    recovered_model_grav = inv3.run(m0)
    mesh.write_model_UBC("CaMP_gravity_synthetic_inversion_model_jw.ubc", recovered_model_grav)

def run_mag_inverison():
    #Inversion
    actvMap = maps.IdentityMap(mesh)

    # mag problem
    simulation_mag = pf.magnetics.simulation.Simulation3DIntegral(
        survey=data_mag.survey,
        mesh=mesh,
        chiMap=actvMap,
    )
    dmis_mag = data_misfit.L2DataMisfit(data=data_mag, simulation=simulation_mag)

    # Initial Model
    m0 = 1e-4 * np.ones(mesh.nC)

    reg = regularization.Simple(mesh, mapping=actvMap, indActive=np.ones(mesh.n_cells, dtype=bool))

    opt = optimization.ProjectedGNCG(
        maxIter=20, lower=0.0, upper=1.0, maxIterLS=20, maxIterCG=20, tolCG=1e-3
    )

    inv_prob = inverse_problem.BaseInvProblem(dmis_mag, reg, opt)

    # Defining a starting value for the trade-off parameter (beta) between the data
    # misfit and the regularization.
    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e-2)
    beta_schedule = directives.BetaSchedule(coolingFactor=5, coolingRate=1)
    update_jacobi = directives.UpdatePreconditioner()
    target_misfit = directives.TargetMisfit(chifact=1)

    # Add sensitivity weights
    sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)

    # The directives are defined as a list.
    directives_list = [
        sensitivity_weights,
        starting_beta,
        beta_schedule,
        update_jacobi,
        target_misfit,
    ]

    inv = inversion.BaseInversion(inv_prob, directives_list)

    # Run inversion
    recovered_model_mag = inv.run(m0)

    mesh.write_model_UBC("CaMP_magnetic_synthetic_inversion_model_jw.ubc", recovered_model_mag)

run_grav_inversion()
run_mag_inverison()