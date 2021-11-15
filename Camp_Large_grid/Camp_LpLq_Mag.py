import discretize as ds
import SimPEG.potential_fields as pf
from SimPEG import (
    maps, utils, simulation, inverse_problem, inversion, optimization, regularization, data_misfit, directives
)
from SimPEG.utils import io_utils
import numpy as np
import matplotlib.pyplot as plt

mesh = ds.TreeMesh.read_UBC('mesh_CaMP_jw.ubc')

data_mag = io_utils.read_mag3d_ubc('magnetic_data_jw.obs')


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

    opt = optimization.ProjectedGNCG(
        maxIter=50, lower=0.0, upper=1.0, maxIterLS=20, maxIterCG=100, tolCG=1e-3
    )

    inv_prob = inverse_problem.BaseInvProblem(dmis_mag, reg, opt)

    # Defining a starting value for the trade-off parameter (beta) between the data
    # misfit and the regularization.
    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e-2)
    beta_schedule = directives.BetaSchedule(coolingFactor=5, coolingRate=1)
    update_jacobi = directives.UpdatePreconditioner()



    # Add sensitivity weights
    sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)
    IRLS = directives.Update_IRLS(f_min_change=1e-4, max_irls_iterations=50, beta_tol=1e-2)

    # The directives are defined as a list.
    directives_list = [
        sensitivity_weights,
        starting_beta,
        beta_schedule,
        IRLS,
        update_jacobi,

    ]

    inv = inversion.BaseInversion(inv_prob, directives_list)

    # Run inversion
    recovered_model_mag = inv.run(m0)

    mesh.write_model_UBC("CaMP_magnetic_synthetic_inversion_model_LPLQ_jw.ubc", recovered_model_mag)

run_mag_inverison()