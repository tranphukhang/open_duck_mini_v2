from pathlib import Path
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer


# ============================================================
# PATH SETUP
# ============================================================

SIMULATION_DIR = Path(__file__).resolve().parent
MPC_SRBD_DIR = SIMULATION_DIR.parent
REPO_ROOT = MPC_SRBD_DIR.parent

STATE_DIR = MPC_SRBD_DIR / "State"
SRBD_DIR = MPC_SRBD_DIR / "SRBD"
QP_DIR = MPC_SRBD_DIR / "QP"
KINEMATICS_DIR = MPC_SRBD_DIR / "Kinematics"
ACTUATOR_DIR = MPC_SRBD_DIR / "Actuator"
COMMAND_DIR = MPC_SRBD_DIR / "Command"


for directory in (
    STATE_DIR,
    SRBD_DIR,
    QP_DIR,
    KINEMATICS_DIR,
    ACTUATOR_DIR,
    COMMAND_DIR
):
    if str(directory) not in sys.path:
        sys.path.insert(
            0,
            str(directory)
        )


# ============================================================
# SIMULATION SETUP
# ============================================================

from simulation_setup import (
    initialize_simulation
)


# ============================================================
# STATE
# ============================================================

from get_current_state import (
    get_current_state
)

from W_euler_base import (
    get_Ew_inverse
)

from W_moment_arm import (
    get_W_moment_arm
)

from W_R_6 import (
    get_W_R_6
)


# ============================================================
# SRBD
# ============================================================

from build_SRBD_model import (
    build_SRBD_model
)


# ============================================================
# QP
# ============================================================

from QP_parameters import (
    N,
    dt_mpc,
    dt_mpc_update
)

from QP_problem import (
    build_QP_problem
)

from QP_diagnostics import (
    print_QP_diagnostics
)


# ============================================================
# KINEMATICS
# ============================================================

from W_wrench_to_joint_torque import (
    get_joint_torque_from_MPC_wrench
)


# ============================================================
# ACTUATOR
# ============================================================

from actuator_mapping import (
    apply_joint_torque
)


# ============================================================
# DISPLAY
# ============================================================

np.set_printoptions(
    precision=6,
    suppress=True,
    linewidth=200
)


# ============================================================
# SIMULATION PARAMETERS
# ============================================================

SIM_DURATION = 30.0

PRINT_EVERY_N_MPC = 10


# ============================================================
# XML
# ============================================================

XML_PATH = (
    REPO_ROOT
    / "xmls"
    / "scene_flat_terrian_v3.xml"
)


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. COMMON SIMULATION SETUP
    # ========================================================

    setup = initialize_simulation(
        xml_path=XML_PATH,
        horizon=N,
        dt_mpc=dt_mpc,
        dt_mpc_update=dt_mpc_update
    )


    model = setup[
        "model"
    ]

    data = setup[
        "data"
    ]

    steps_per_mpc_update = setup[
        "steps_per_mpc_update"
    ]

    mass_CoM = setup[
        "mass_CoM"
    ]

    W_inertia_CoM_q0 = setup[
        "W_inertia_CoM_q0"
    ]

    reference_generator = setup[
        "reference_generator"
    ]

    qp_solver = setup[
        "qp_solver"
    ]


    # ========================================================
    # 2. STANDING COMMAND
    # ========================================================

    v_CoM_cmd = np.array(
        [
            0.0,
            0.0,
            0.0
        ],
        dtype=float
    )


    print("\n")
    print("=" * 70)
    print("STANDING COMMAND")
    print("=" * 70)


    print(
        "v_CoM_cmd [m/s] ="
    )


    print(
        v_CoM_cmd
    )


    # ========================================================
    # 3. LOGGING
    # ========================================================

    log_time = []

    log_x = []

    log_u0 = []

    log_tau = []

    log_mpc_compute_time = []

    log_qp_solve_time = []


    # ========================================================
    # 4. CLOSED-LOOP SIMULATION
    # ========================================================

    mpc_iteration = 0


    print("\n")
    print("=" * 70)
    print("START CLOSED-LOOP MPC STANDING TEST")
    print("=" * 70)


    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:


        while (
            viewer.is_running()
            and
            data.time < SIM_DURATION
        ):


            # =================================================
            # CURRENT CONTROL CYCLE
            # =================================================

            mpc_compute_start = (
                time.perf_counter()
            )


            # =================================================
            # STEP 1:
            #
            # MEASURE x_k
            # =================================================

            x_k = get_current_state(
                model,
                data
            )


            # =================================================
            # STEP 2:
            #
            # GENERATE MPC REFERENCE
            # =================================================

            X_ref = (
                reference_generator.generate_reference(
                    v_CoM_cmd
                )
            )


            # =================================================
            # STEP 3:
            #
            # UPDATE NONLINEAR QUANTITIES AT x_k
            # =================================================

            W_euler_base = x_k[
                3:6
            ]


            Ew_inverse = get_Ew_inverse(
                W_euler_base
            )


            (
                W_r_L,
                W_r_R
            ) = get_W_moment_arm(
                model,
                data
            )


            (
                L6_R_W,
                R6_R_W
            ) = get_W_R_6(
                model,
                data
            )


            # =================================================
            # STEP 4:
            #
            # BUILD SRBD PREDICTION MODEL
            #
            #     continuous
            #         ↓
            #     discretization
            #         ↓
            #     prediction
            #
            #     X = Phi x_k + Gamma U
            # =================================================

            (
                Phi,
                Gamma
            ) = build_SRBD_model(

                Ew_inverse=Ew_inverse,

                mass_CoM=mass_CoM,

                W_inertia_CoM=(
                    W_inertia_CoM_q0
                ),

                W_r_L=W_r_L,

                W_r_R=W_r_R,

                dt_mpc=dt_mpc,

                horizon=N
            )


            # =================================================
            # STEP 5:
            #
            # BUILD QP
            # =================================================

            (
                H,
                f,
                A,
                l,
                u
            ) = build_QP_problem(

                Phi=Phi,

                Gamma=Gamma,

                x_current=x_k,

                X_ref=X_ref,

                N=N,

                L6_R_W=L6_R_W,

                R6_R_W=R6_R_W
            )


            # =================================================
            # STEP 6:
            #
            # SETUP AND SOLVE QP
            # =================================================

            qp_solver.setup(
                H,
                f,
                A,
                l,
                u,
                print_diagnostics=False
            )


            (
                U_opt,
                u0_opt,
                solver_info,
                post_diagnostics
            ) = qp_solver.solve(
                print_diagnostics=False
            )


            # =================================================
            # STEP 7:
            #
            # CHECK MPC SOLUTION
            # =================================================

            if u0_opt is None:

                data.ctrl[:] = 0.0


                print("\n")
                print("=" * 70)
                print("MPC STOPPED")
                print("=" * 70)


                print(
                    "simulation time =",
                    data.time,
                    "s"
                )


                print(
                    "MPC iteration =",
                    mpc_iteration
                )


                if post_diagnostics is not None:

                    print_QP_diagnostics(
                        post_diagnostics
                    )


                else:

                    print(
                        "\nOSQP solver information:"
                    )


                    print(
                        "status =",
                        solver_info.get(
                            "status"
                        )
                    )


                    print(
                        "status_val =",
                        solver_info.get(
                            "status_val"
                        )
                    )


                    print(
                        "iterations =",
                        solver_info.get(
                            "iterations"
                        )
                    )


                    print(
                        "primal_residual =",
                        solver_info.get(
                            "primal_residual"
                        )
                    )


                    print(
                        "dual_residual =",
                        solver_info.get(
                            "dual_residual"
                        )
                    )


                    print(
                        "solve_time =",
                        solver_info.get(
                            "solve_time"
                        ),
                        "s"
                    )


                raise RuntimeError(
                    "Không có MPC input hợp lệ để áp dụng."
                )


            # =================================================
            # STEP 8:
            #
            # CURRENT MPC INPUT
            # =================================================

            u_k = u0_opt.copy()


            # =================================================
            # STEP 9:
            #
            # WRENCH -> JOINT TORQUE
            # =================================================

            (
                tau_k,
                tau_L,
                tau_R
            ) = get_joint_torque_from_MPC_wrench(
                model,
                data,
                u_k
            )


            # =================================================
            # STEP 10:
            #
            # APPLY JOINT TORQUE
            # =================================================

            apply_joint_torque(
                model,
                data,
                tau_k
            )


            # =================================================
            # MPC COMPUTATION TIME
            # =================================================

            mpc_compute_time = (
                time.perf_counter()
                -
                mpc_compute_start
            )


            # =================================================
            # LOG CURRENT CONTROL CYCLE
            # =================================================

            log_time.append(
                float(
                    data.time
                )
            )


            log_x.append(
                x_k.copy()
            )


            log_u0.append(
                u_k.copy()
            )


            log_tau.append(
                tau_k.copy()
            )


            log_mpc_compute_time.append(
                mpc_compute_time
            )


            log_qp_solve_time.append(
                solver_info[
                    "solve_time"
                ]
            )


            # =================================================
            # STEP 11:
            #
            # PROPAGATE MUJOCO PLANT
            # =================================================

            for _ in range(
                steps_per_mpc_update
            ):

                if data.time >= SIM_DURATION:

                    break


                mujoco.mj_step(
                    model,
                    data
                )


            # =================================================
            # STEP 12:
            #
            # MEASURE x_k+1
            # =================================================

            x_k_plus_1 = get_current_state(
                model,
                data
            )


            # =================================================
            # STATUS PRINT
            # =================================================

            if (
                mpc_iteration
                %
                PRINT_EVERY_N_MPC
                ==
                0
            ):

                print("\n")
                print("-" * 70)


                print(
                    "MPC iteration =",
                    mpc_iteration
                )


                print(
                    "simulation time =",
                    f"{data.time:.4f}",
                    "s"
                )


                print(
                    "\nCoM position reference ="
                )


                print(
                    reference_generator.get_position_reference()
                )


                print(
                    "\nx_k CoM position ="
                )


                print(
                    x_k[
                        0:3
                    ]
                )


                print(
                    "x_k+1 CoM position ="
                )


                print(
                    x_k_plus_1[
                        0:3
                    ]
                )


                print(
                    "\nx_k Euler [rad] ="
                )


                print(
                    x_k[
                        3:6
                    ]
                )


                print(
                    "x_k+1 Euler [rad] ="
                )


                print(
                    x_k_plus_1[
                        3:6
                    ]
                )


                print(
                    "\nu_k ="
                )


                print(
                    u_k
                )


                print(
                    "\ntau_k ="
                )


                print(
                    tau_k
                )


                print(
                    "\nOSQP status =",
                    solver_info[
                        "status"
                    ]
                )


                print(
                    "OSQP iterations =",
                    solver_info[
                        "iterations"
                    ]
                )


                print(
                    "OSQP primal residual =",
                    solver_info[
                        "primal_residual"
                    ]
                )


                print(
                    "OSQP dual residual =",
                    solver_info[
                        "dual_residual"
                    ]
                )


                print(
                    "OSQP solve time =",
                    solver_info[
                        "solve_time"
                    ],
                    "s"
                )


                if post_diagnostics is not None:

                    print(
                        "Constraint max violation =",
                        post_diagnostics[
                            "max_violation"
                        ]
                    )


                print(
                    "Full MPC compute time =",
                    mpc_compute_time,
                    "s"
                )


            # =================================================
            # VIEWER
            # =================================================

            viewer.sync()


            # =================================================
            # NEXT MPC CYCLE
            # =================================================

            mpc_iteration += 1


    # ========================================================
    # 5. CONVERT LOGS
    # ========================================================

    log_time = np.asarray(
        log_time,
        dtype=float
    )


    log_x = np.asarray(
        log_x,
        dtype=float
    )


    log_u0 = np.asarray(
        log_u0,
        dtype=float
    )


    log_tau = np.asarray(
        log_tau,
        dtype=float
    )


    log_mpc_compute_time = np.asarray(
        log_mpc_compute_time,
        dtype=float
    )


    log_qp_solve_time = np.asarray(
        log_qp_solve_time,
        dtype=float
    )


    # ========================================================
    # 6. FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("SIMULATION FINISHED")
    print("=" * 70)


    print(
        "MPC iterations =",
        mpc_iteration
    )


    if log_mpc_compute_time.size > 0:

        print(
            "Average MPC compute time =",
            np.mean(
                log_mpc_compute_time
            ),
            "s"
        )


        print(
            "Maximum MPC compute time =",
            np.max(
                log_mpc_compute_time
            ),
            "s"
        )


    if log_qp_solve_time.size > 0:

        print(
            "Average OSQP solve time =",
            np.mean(
                log_qp_solve_time
            ),
            "s"
        )


    if log_x.shape[0] > 0:

        print(
            "\nInitial logged CoM ="
        )


        print(
            log_x[
                0,
                0:3
            ]
        )


        print(
            "Final logged CoM ="
        )


        print(
            log_x[
                -1,
                0:3
            ]
        )


    print(
        "\nFinal CoM position reference ="
    )


    print(
        reference_generator.get_position_reference()
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()