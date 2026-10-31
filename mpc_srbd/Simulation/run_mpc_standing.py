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
# STATE
# ============================================================

from W_position_CoM import (
    get_W_position_CoM
)

from W_linear_velocity_CoM import (
    get_W_linear_velocity_CoM
)

from W_angular_velocity_base import (
    get_W_angular_velocity_base
)

from W_euler_base import (
    get_W_euler_base,
    get_Ew_inverse
)

from mass_CoM import (
    get_mass_CoM
)

from W_inertia_CoM import (
    get_W_inertia_CoM
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

from continuous_model import (
    build_continuous_model
)

from discretization import (
    discretize_model
)

from prediction_model import (
    build_prediction_model
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

from QP_solver import (
    QPSolver
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
# COMMAND
# ============================================================

from reference_generator import (
    ReferenceGenerator
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
# READ CURRENT STATE
# ============================================================

def get_current_state(
    model,
    data
):
    """
    Read current SRBD state x_k.

    State:

        x =
        [
            p_CoM,
            euler_base,
            v_CoM,
            omega_base,
            1
        ]

    Important:

        mj_forward()

    does NOT advance simulation time.

    Only:

        mj_step()

    advances the plant.
    """

    # --------------------------------------------------------
    # Synchronize MuJoCo derived quantities
    # --------------------------------------------------------

    mujoco.mj_forward(
        model,
        data
    )


    # --------------------------------------------------------
    # Update subtree CoM velocity
    # --------------------------------------------------------

    mujoco.mj_subtreeVel(
        model,
        data
    )


    # --------------------------------------------------------
    # Read state components
    # --------------------------------------------------------

    W_position_CoM = (
        get_W_position_CoM(
            model,
            data
        )
    )


    W_euler_base = (
        get_W_euler_base(
            model,
            data
        )
    )


    W_linear_velocity_CoM = (
        get_W_linear_velocity_CoM(
            model,
            data
        )
    )


    W_angular_velocity_base = (
        get_W_angular_velocity_base(
            model,
            data
        )
    )


    # --------------------------------------------------------
    # Assemble augmented state
    # --------------------------------------------------------

    x_current = np.concatenate([
        W_position_CoM,
        W_euler_base,
        W_linear_velocity_CoM,
        W_angular_velocity_base,
        np.array([1.0])
    ])


    if x_current.shape != (13,):

        raise RuntimeError(
            "x_current phải có shape (13,), "
            f"nhưng nhận được {x_current.shape}."
        )


    if not np.all(
        np.isfinite(
            x_current
        )
    ):

        raise RuntimeError(
            "x_current chứa NaN hoặc Inf."
        )


    return x_current


# ============================================================
# PRINT QP FAILURE
# ============================================================

def print_QP_failure(
    data,
    mpc_iteration,
    solver_info,
    post_diagnostics
):
    """
    Print detailed information when OSQP reports a solution
    but post-solve diagnostics reject it.
    """

    print("\n")
    print("=" * 70)
    print("QP SOLUTION REJECTED")
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


    print("\nSolver information:")


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
        "setup_time =",
        solver_info.get(
            "setup_time"
        )
    )


    print(
        "solve_time =",
        solver_info.get(
            "solve_time"
        )
    )


    print(
        "run_time =",
        solver_info.get(
            "run_time"
        )
    )


    # --------------------------------------------------------
    # Post diagnostics
    # --------------------------------------------------------

    if post_diagnostics is None:

        print(
            "\npost_diagnostics = None"
        )

        return


    print("\nPost-solve diagnostics:")


    keys_to_print = (
        "solution_dimension_ok",
        "solution_finite",
        "objective_value",
        "objective_finite",
        "max_lower_violation",
        "max_upper_violation",
        "max_violation",
        "constraint_ok",
        "solver_status",
        "solver_status_val",
        "solution_ok",
        "errors",
        "warnings"
    )


    for key in keys_to_print:

        print(
            f"{key} =",
            post_diagnostics.get(
                key
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. LOAD MODEL
    # ========================================================

    if not XML_PATH.exists():

        raise FileNotFoundError(
            f"Không tìm thấy XML:\n{XML_PATH}"
        )


    model = mujoco.MjModel.from_xml_path(
        str(XML_PATH)
    )


    data = mujoco.MjData(
        model
    )


    # ========================================================
    # 2. SIMULATION TIMING
    # ========================================================

    dt_sim = float(
        model.opt.timestep
    )


    steps_per_mpc_update = int(
        round(
            dt_mpc_update
            /
            dt_sim
        )
    )


    # --------------------------------------------------------
    # dt_mpc_update must be an integer number of MuJoCo steps
    # --------------------------------------------------------

    if not np.isclose(
        steps_per_mpc_update
        *
        dt_sim,
        dt_mpc_update,
        atol=1e-12
    ):

        raise ValueError(
            "dt_mpc_update phải là bội số nguyên "
            "của dt_sim."
        )


    print("\n")
    print("=" * 70)
    print("SIMULATION TIMING")
    print("=" * 70)


    print(
        "dt_sim            =",
        dt_sim,
        "s"
    )


    print(
        "dt_mpc            =",
        dt_mpc,
        "s"
    )


    print(
        "dt_mpc_update     =",
        dt_mpc_update,
        "s"
    )


    print(
        "MPC update rate   =",
        1.0 / dt_mpc_update,
        "Hz"
    )


    print(
        "MuJoCo steps/MPC  =",
        steps_per_mpc_update
    )


    print(
        "Prediction horizon =",
        N * dt_mpc,
        "s"
    )


    # ========================================================
    # 3. INITIALIZE MUJOCO FROM HOME KEYFRAME
    # ========================================================

    home_key_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home"
    )


    if home_key_id < 0:

        raise RuntimeError(
            'Không tìm thấy keyframe "home" trong XML.'
        )


    # --------------------------------------------------------
    # Reset MuJoCo state to keyframe "home"
    # --------------------------------------------------------

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_key_id
    )


    # --------------------------------------------------------
    # Update kinematics/dynamics at the home configuration
    #
    # mj_forward does NOT advance simulation time.
    # --------------------------------------------------------

    mujoco.mj_forward(
        model,
        data
    )


    mujoco.mj_subtreeVel(
        model,
        data
    )


    # ========================================================
    # 4. CONSTANT SRBD PARAMETERS
    # ========================================================

    mass_CoM = get_mass_CoM(
        model
    )


    # --------------------------------------------------------
    # Freeze ICoM at initial configuration q0
    # --------------------------------------------------------

    W_inertia_CoM_q0 = (
        get_W_inertia_CoM(
            model,
            data
        )
    )


    print("\n")
    print("=" * 70)
    print("FROZEN SRBD PARAMETERS")
    print("=" * 70)


    print(
        "mass_CoM =",
        mass_CoM,
        "kg"
    )


    print(
        "\nW_inertia_CoM(q0) ="
    )

    print(
        W_inertia_CoM_q0
    )


    # ========================================================
    # 5. INITIAL STATE AND COMMAND
    # ========================================================

    x_initial = get_current_state(
        model,
        data
    )


    # --------------------------------------------------------
    # Desired CoM linear velocity command [m/s]
    #
    # Current standing test:
    #
    #     vx_cmd = 0
    #     vy_cmd = 0
    #     vz_cmd = 0
    # --------------------------------------------------------

    v_CoM_cmd = np.array(
        [
            0.0,
            0.0,
            0.0
        ],
        dtype=float
    )


    # --------------------------------------------------------
    # Initialize reference generator
    #
    # The generator stores its own CoM position reference.
    #
    # Therefore:
    #
    #     p_ref
    #
    # is independent of the measured robot state x_k.
    #
    # Prediction grid:
    #
    #     dt_prediction = dt_mpc
    #
    # Reference update:
    #
    #     dt_update = dt_mpc_update
    # --------------------------------------------------------

    reference_generator = ReferenceGenerator(
        x_initial=x_initial,
        horizon=N,
        dt_prediction=dt_mpc,
        dt_update=dt_mpc_update
    )


    print("\n")
    print("=" * 70)
    print("INITIAL STATE AND COMMAND")
    print("=" * 70)


    print(
        "x_0 ="
    )

    print(
        x_initial
    )


    print(
        "\nv_CoM_cmd [m/s] ="
    )

    print(
        v_CoM_cmd
    )


    print(
        "\nInitial CoM position reference ="
    )

    print(
        reference_generator.get_position_reference()
    )


    # ========================================================
    # 6. QP SOLVER
    #
    # For first integration test:
    #
    # - rebuild QP every cycle
    # - setup OSQP every cycle
    # - no warm start yet
    #
    # This is intentionally not optimized for real-time.
    # ========================================================

    qp_solver = QPSolver(
        verbose=False,
        eps_abs=1e-5,
        eps_rel=1e-5,
        max_iter=1000,
        polishing=True,
        enable_warm_start=False,
        constraint_tolerance=1e-4
    )


    # ========================================================
    # 7. LOGGING
    # ========================================================

    log_time = []

    log_x = []

    log_u0 = []

    log_tau = []

    log_mpc_compute_time = []

    log_qp_solve_time = []


    # ========================================================
    # 8. CLOSED-LOOP SIMULATION
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
            #
            # At this instant:
            #
            #                x_k
            #
            # Control sequence:
            #
            # x_k
            #  |
            #  v
            # Command / Reference Generator
            #  |
            #  v
            # X_ref
            #  |
            #  v
            # MPC
            #  |
            #  v
            # u_k
            #  |
            #  v
            # tau_k
            #  |
            #  v
            # MuJoCo plant
            #  |
            #  v
            # x_k+1
            #
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
            #
            # Input:
            #
            #     v_CoM_cmd
            #
            # Output:
            #
            #     X_ref
            #
            # ReferenceGenerator maintains p_ref internally.
            #
            # Therefore:
            #
            #     p_ref != measured p_CoM
            #
            # in general.
            #
            # For current standing command:
            #
            #     v_CoM_cmd = [0, 0, 0]
            #
            # p_ref remains equal to the initial home CoM
            # position.
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
            #
            # These values are frozen over the current
            # MPC prediction horizon.
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
            # CONTINUOUS SRBD
            #
            # x_dot = Ac x + Bc u
            # =================================================

            (
                Ac,
                Bc
            ) = build_continuous_model(

                Ew_inverse=Ew_inverse,

                mass_CoM=mass_CoM,

                W_inertia_CoM=(
                    W_inertia_CoM_q0
                ),

                W_r_L=W_r_L,

                W_r_R=W_r_R
            )


            # =================================================
            # STEP 5:
            #
            # MPC DISCRETIZATION
            #
            # IMPORTANT:
            #
            #     dt_mpc = prediction-grid spacing
            #
            # Current:
            #
            #     dt_mpc = 0.025 s
            # =================================================

            (
                Ad,
                Bd
            ) = discretize_model(
                Ac,
                Bc,
                dt_mpc
            )


            # =================================================
            # STEP 6:
            #
            # PREDICTION MODEL
            #
            # X =
            #
            # Phi x_k
            #
            # +
            #
            # Gamma U
            # =================================================

            (
                Phi,
                Gamma
            ) = build_prediction_model(
                Ad,
                Bd,
                N
            )


            # =================================================
            # STEP 7:
            #
            # BUILD QP USING CURRENT x_k
            #
            # Tracking reference:
            #
            #     X_ref
            #
            # is provided by ReferenceGenerator.
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
            # STEP 8:
            #
            # SOLVE MPC
            #
            # U* =
            #
            # [
            #     u_k
            #     u_k+1
            #     ...
            #     u_k+N-1
            # ]
            #
            # Receding-horizon control:
            #
            # only:
            #
            #     u_k = u0_opt
            #
            # is applied to the plant.
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
            # STEP 9:
            #
            # CHECK MPC SOLUTION
            #
            # Important:
            #
            # OSQP status = solved
            #
            # does NOT necessarily mean:
            #
            #     u0_opt != None
            #
            # because post-solve diagnostics may reject
            # the solution.
            # =================================================

            if u0_opt is None:

                # Remove torque before stopping

                data.ctrl[:] = 0.0


                print_QP_failure(
                    data=data,
                    mpc_iteration=mpc_iteration,
                    solver_info=solver_info,
                    post_diagnostics=(
                        post_diagnostics
                    )
                )


                raise RuntimeError(
                    "QP solution bị "
                    "post-solve diagnostics từ chối."
                )


            # =================================================
            # STEP 10:
            #
            # CURRENT MPC INPUT
            #
            # u_k =
            #
            # [
            #     F_L,
            #     M_L,
            #     F_R,
            #     M_R
            # ]
            #
            # Dimension:
            #
            #     u_k in R^12
            # =================================================

            u_k = u0_opt.copy()


            # =================================================
            # STEP 11:
            #
            # WRENCH -> JOINT TORQUE
            #
            # u_k
            #
            #    ->
            #
            # tau_k
            #
            # Dimension:
            #
            #     tau_k in R^10
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
            # STEP 12:
            #
            # APPLY tau_k TO data.ctrl
            #
            # IMPORTANT:
            #
            # The plant is STILL at x_k.
            #
            # We have NOT called mj_step() yet.
            # =================================================

            apply_joint_torque(
                model,
                data,
                tau_k
            )


            # =================================================
            # MPC COMPUTATION TIME
            #
            # For now this is only monitoring.
            #
            # It is NOT required to be < dt_mpc_update during
            # this development/debugging stage.
            # =================================================

            mpc_compute_time = (
                time.perf_counter()
                -
                mpc_compute_start
            )


            # =================================================
            # LOG:
            #
            # x_k and u_k before plant propagation
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
            # STEP 13:
            #
            # PROPAGATE MUJOCO PLANT
            #
            # This is the ONLY part that advances time.
            #
            # tau_k is held constant using ZOH.
            #
            # Current:
            #
            #     dt_mpc_update = 0.01 s
            #
            #     dt_sim = 0.0005 s
            #
            # therefore:
            #
            #     20 MuJoCo steps / MPC update
            #
            #
            # x_k
            #   |
            #   | u_k / tau_k
            #   v
            # mj_step()
            # mj_step()
            # ...
            # mj_step()
            #   |
            #   v
            # x_k+1
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
            # STEP 14:
            #
            # MEASURE x_k+1
            #
            # This state is measured AFTER u_k has acted
            # on the MuJoCo plant.
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


                # --------------------------------------------
                # Current CoM reference
                # --------------------------------------------

                print(
                    "\nCoM position reference ="
                )

                print(
                    reference_generator.get_position_reference()
                )


                # --------------------------------------------
                # x_k -> x_k+1
                # --------------------------------------------

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


                # --------------------------------------------
                # Current MPC wrench
                # --------------------------------------------

                print(
                    "\nu_k ="
                )

                print(
                    u_k
                )


                # --------------------------------------------
                # Current joint torque
                # --------------------------------------------

                print(
                    "\ntau_k ="
                )

                print(
                    tau_k
                )


                # --------------------------------------------
                # Solver
                # --------------------------------------------

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


                # --------------------------------------------
                # Post diagnostics
                # --------------------------------------------

                if post_diagnostics is not None:

                    print(
                        "Constraint max violation =",
                        post_diagnostics[
                            "max_violation"
                        ]
                    )


                # --------------------------------------------
                # Total computation time
                # --------------------------------------------

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
            # NEXT CONTROL CYCLE
            #
            # Current MuJoCo state:
            #
            #     x_k+1
            #
            # Next loop:
            #
            #     x_k <- x_k+1
            #
            # while the command layer independently updates:
            #
            #     p_ref,k
            #
            # according to:
            #
            #     v_CoM_cmd
            # =================================================

            mpc_iteration += 1


    # ========================================================
    # 9. CONVERT LOGS
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
    # 10. FINAL SUMMARY
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