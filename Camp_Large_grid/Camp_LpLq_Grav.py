import discretize as ds
import SimPEG.potential_fields as pf
from SimPEG import (
    maps, utils, simulation, inverse_problem, inversion, optimization, regularization, data_misfit, directives
)
from SimPEG.utils import io_utils
import numpy as np
import matplotlib.pyplot as plt

mesh = ds.TreeMesh.read_UBC('mesh_CaMP_jw.ubc')

data_grav = io_utils.read_grav3d_ubc('grav_data_jw.obs')


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
    reg = regularization.Sparse(
        mesh,
        indActive=np.ones(mesh.n_cells, dtype=bool),
        mapping=actvMap,
        alpha_s=1.0,
        alpha_x=1.0,
        alpha_y=1.0,
        alpha_z=1.0,
        gradientType='total'
    )
    reg.norms = [[0, 1, 1, 1]]

    # Define how the optimization problem is solved. Here we will use a projected
    # Gauss-Newton approach that employs the conjugate gradient solver.
    opt = optimization.ProjectedGNCG(
        maxIter=50, lower=-10, upper=10, maxIterLS=20, maxIterCG=20, tolCG=1e-4
    )

    # Here we define the inverse problem that is to be solved
    inv_prob = inverse_problem.BaseInvProblem(dmis_grav, reg, opt)

    # Defining a starting value for the trade-off parameter (beta) between the data
    # misfit and the regularization.
    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e0)
    beta_schedule = directives.BetaSchedule(coolingFactor=5, coolingRate=1)
    update_jacobi = directives.UpdatePreconditioner()
    Save = directives.SaveOutputEveryIteration(save_txt = True)

    # Add sensitivity weights
    sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)

    IRLS = directives.Update_IRLS(f_min_change=1e-3, max_irls_iterations=50, beta_tol=5e-1)

    # The directives are defined as a list.
    directives_list = [
        Save,
        sensitivity_weights,
        starting_beta,
        beta_schedule,
        IRLS,
        update_jacobi
    ]

    inv3 = inversion.BaseInversion(inv_prob, directives_list)


    recovered_model_grav = inv3.run(m0)
    #Save.plot_misfit_curves()
    #Save.plot_tikhonov_curves()
    mesh.write_model_UBC("CaMP_gravity_synthetic_inversion_model_LPLQ_jw.ubc", recovered_model_grav)

run_grav_inversion()