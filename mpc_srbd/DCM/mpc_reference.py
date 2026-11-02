from __future__ import annotations

import numpy as np

from Command.contact_schedule import (
    PHASE_LSS,
    PHASE_DS_L_TO_R,
    PHASE_RSS,
    PHASE_DS_R_TO_L,
    VALID_PHASES,
)


# ============================================================
# VALIDATION
# ============================================================

def _as_vector3(value, name: str) -> np.ndarray:
    vector = np.asarray(
        value,
        dtype=float,
    ).reshape(3)

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"{name} chứa NaN hoặc Inf."
        )

    return vector.copy()


def _as_plan(value, horizon: int, name: str) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=float,
    )

    if array.shape != (
        horizon,
        3,
    ):
        raise ValueError(
            f"{name} phải có shape ({horizon}, 3), "
            f"nhưng nhận {array.shape}."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} chứa NaN hoặc Inf."
        )

    return array.copy()


def _positive_scalar(value, name: str) -> float:
    value = float(value)

    if not np.isfinite(value):
        raise ValueError(
            f"{name} phải hữu hạn."
        )

    if value <= 0.0:
        raise ValueError(
            f"{name} phải > 0."
        )

    return value


def _validate_phase_preview(
    phase_preview: dict,
    horizon: int,
):
    required_keys = (
        "phase_sequence",
        "phase_progress_sequence",
        "contact_schedule",
    )

    for key in required_keys:
        if key not in phase_preview:
            raise KeyError(
                f"phase_preview thiếu khóa {key!r}."
            )

    phase_sequence = list(
        phase_preview[
            "phase_sequence"
        ]
    )

    if len(phase_sequence) != horizon:
        raise ValueError(
            "phase_sequence length phải bằng horizon."
        )

    for phase in phase_sequence:
        if phase not in VALID_PHASES:
            raise ValueError(
                f"Unsupported phase in preview: {phase!r}."
            )

    progress = np.asarray(
        phase_preview[
            "phase_progress_sequence"
        ],
        dtype=float,
    ).reshape(-1)

    if progress.shape != (
        horizon,
    ):
        raise ValueError(
            "phase_progress_sequence sai shape."
        )

    if not np.all(np.isfinite(progress)):
        raise ValueError(
            "phase_progress_sequence chứa NaN hoặc Inf."
        )

    if np.any(
        progress < -1e-12
    ) or np.any(
        progress > 1.0 + 1e-12
    ):
        raise ValueError(
            "phase_progress_sequence phải nằm trong [0,1]."
        )

    contact_schedule = np.asarray(
        phase_preview[
            "contact_schedule"
        ],
        dtype=int,
    )

    if contact_schedule.shape != (
        horizon,
        2,
    ):
        raise ValueError(
            "contact_schedule sai shape."
        )

    return (
        phase_sequence,
        np.clip(
            progress,
            0.0,
            1.0,
        ),
        contact_schedule.copy(),
    )


# ============================================================
# SUPPORT / VRP REFERENCE
# ============================================================

def build_support_reference_from_phase_preview(
    phase_preview: dict,
    W_p_L6_plan,
    W_p_R6_plan,
    horizon: int,
):
    """
    Build support/VRP reference from the SAME directional phase preview
    used by the contact schedule and foot-plan horizon.

    Directional phase semantics
    ---------------------------
    LSS:
        r = p_L

    RSS:
        r = p_R

    DS_L_TO_R:
        r(s) = (1-s) p_L + s p_R

    DS_R_TO_L:
        r(s) = (1-s) p_R + s p_L

    where s is the normalized phase progress in [0,1].
    """
    horizon = int(
        horizon
    )

    if horizon <= 0:
        raise ValueError(
            "horizon phải > 0."
        )

    (
        phase_sequence,
        phase_progress,
        contact_schedule,
    ) = _validate_phase_preview(
        phase_preview=phase_preview,
        horizon=horizon,
    )

    W_p_L6_plan = _as_plan(
        W_p_L6_plan,
        horizon,
        "W_p_L6_plan",
    )

    W_p_R6_plan = _as_plan(
        W_p_R6_plan,
        horizon,
        "W_p_R6_plan",
    )

    W_support_ref = np.zeros(
        (horizon, 3),
        dtype=float,
    )

    for k in range(horizon):
        phase = phase_sequence[k]
        s = float(
            phase_progress[k]
        )

        p_L = W_p_L6_plan[k]
        p_R = W_p_R6_plan[k]

        if phase == PHASE_LSS:
            W_support_ref[k] = p_L

        elif phase == PHASE_RSS:
            W_support_ref[k] = p_R

        elif phase == PHASE_DS_L_TO_R:
            W_support_ref[k] = (
                (1.0 - s) * p_L
                +
                s * p_R
            )

        elif phase == PHASE_DS_R_TO_L:
            W_support_ref[k] = (
                (1.0 - s) * p_R
                +
                s * p_L
            )

        else:
            raise RuntimeError(
                f"Unhandled phase: {phase!r}"
            )

    return {
        "W_support_ref": W_support_ref,
        "phase_sequence": phase_sequence,
        "phase_progress_sequence": phase_progress.copy(),
        "contact_schedule": contact_schedule,
    }


# ============================================================
# DCM BACKWARD RECURSION
# ============================================================

def compute_nominal_dcm_horizon(
    W_support_ref,
    omega: float,
    dt_mpc: float,
    terminal_velocity_xy=None,
):
    """
    Compute the nominal DCM horizon.

    DCM dynamics for constant support r over one prediction interval:

        xi_dot = omega (xi - r)

    Backward recursion:

        xi_k
          =
        r_k
          +
        exp(-omega*dt_mpc)
        (xi_{k+1} - r_k)

    Terminal condition:

        xi_N
          =
        r_{N-1}
          +
        v_terminal / omega

    No measured-CoM correction is applied here.
    """
    W_support_ref = np.asarray(
        W_support_ref,
        dtype=float,
    )

    if (
        W_support_ref.ndim != 2
        or W_support_ref.shape[1] != 3
        or W_support_ref.shape[0] <= 0
    ):
        raise ValueError(
            "W_support_ref phải có shape (N,3)."
        )

    if not np.all(
        np.isfinite(
            W_support_ref
        )
    ):
        raise ValueError(
            "W_support_ref chứa NaN hoặc Inf."
        )

    omega = _positive_scalar(
        omega,
        "omega",
    )

    dt_mpc = _positive_scalar(
        dt_mpc,
        "dt_mpc",
    )

    if terminal_velocity_xy is None:
        terminal_velocity_xy = np.zeros(
            2,
            dtype=float,
        )
    else:
        terminal_velocity_xy = np.asarray(
            terminal_velocity_xy,
            dtype=float,
        ).reshape(2)

        if not np.all(
            np.isfinite(
                terminal_velocity_xy
            )
        ):
            raise ValueError(
                "terminal_velocity_xy chứa NaN hoặc Inf."
            )

    N = W_support_ref.shape[0]

    W_xi_ref_xy = np.zeros(
        (N + 1, 2),
        dtype=float,
    )

    W_xi_ref_xy[N] = (
        W_support_ref[-1, 0:2]
        +
        terminal_velocity_xy
        / omega
    )

    alpha = float(
        np.exp(
            -omega
            * dt_mpc
        )
    )

    for k in range(
        N - 1,
        -1,
        -1,
    ):
        r_k = (
            W_support_ref[
                k,
                0:2,
            ]
        )

        W_xi_ref_xy[k] = (
            r_k
            +
            alpha
            * (
                W_xi_ref_xy[k + 1]
                -
                r_k
            )
        )

    return {
        "W_xi_ref_xy": W_xi_ref_xy,
        "alpha_backward": alpha,
    }


# ============================================================
# CoM RECONSTRUCTION FROM NOMINAL DCM
# ============================================================

def reconstruct_nominal_com_reference(
    W_xi_ref_xy,
    W_support_ref,
    W_p_CoM_nominal_current,
    com_height: float,
    omega: float,
    dt_mpc: float,
):
    """
    Reconstruct nominal CoM states from the DCM horizon.

    IMPORTANT
    ---------
    DCM alone does not uniquely determine CoM position. A current nominal
    CoM reference state is therefore required:

        W_p_CoM_nominal_current

    This is a PLANNER/REFERENCE state, not measured robot feedback.

    For one interval with constant support r_k:

        xi(t)
          =
        r_k
          +
        exp(omega t) (xi_k - r_k)

    and:

        c_dot = omega (xi - c)

    The exact CoM solution over dt is:

        c_{k+1}
          =
        r_k
          +
        exp(-omega dt) (c_k - r_k)
          +
        sinh(omega dt) (xi_k - r_k)

    Then:

        v_{k+1}
          =
        omega (xi_{k+1} - c_{k+1})

    Returned rows correspond to future MPC states:

        row 0 -> x_{k+1}
        ...
        row N-1 -> x_{k+N}
    """
    W_xi_ref_xy = np.asarray(
        W_xi_ref_xy,
        dtype=float,
    )

    W_support_ref = np.asarray(
        W_support_ref,
        dtype=float,
    )

    if (
        W_xi_ref_xy.ndim != 2
        or W_xi_ref_xy.shape[1] != 2
        or W_xi_ref_xy.shape[0] < 2
    ):
        raise ValueError(
            "W_xi_ref_xy phải có shape (N+1,2)."
        )

    N = (
        W_xi_ref_xy.shape[0] - 1
    )

    if W_support_ref.shape != (
        N,
        3,
    ):
        raise ValueError(
            f"W_support_ref phải có shape ({N},3)."
        )

    if not np.all(
        np.isfinite(
            W_xi_ref_xy
        )
    ):
        raise ValueError(
            "W_xi_ref_xy chứa NaN hoặc Inf."
        )

    if not np.all(
        np.isfinite(
            W_support_ref
        )
    ):
        raise ValueError(
            "W_support_ref chứa NaN hoặc Inf."
        )

    W_p_CoM_nominal_current = _as_vector3(
        W_p_CoM_nominal_current,
        "W_p_CoM_nominal_current",
    )

    com_height = _positive_scalar(
        com_height,
        "com_height",
    )

    omega = _positive_scalar(
        omega,
        "omega",
    )

    dt_mpc = _positive_scalar(
        dt_mpc,
        "dt_mpc",
    )

    W_p_CoM_ref = np.zeros(
        (N, 3),
        dtype=float,
    )

    W_v_CoM_ref = np.zeros(
        (N, 3),
        dtype=float,
    )

    c_xy = (
        W_p_CoM_nominal_current[
            0:2
        ].copy()
    )

    beta = float(
        np.exp(
            -omega
            * dt_mpc
        )
    )

    sinh_term = float(
        np.sinh(
            omega
            * dt_mpc
        )
    )

    # Nominal velocity at the current reference state.
    W_v_CoM_nominal_current = np.zeros(
        3,
        dtype=float,
    )

    W_v_CoM_nominal_current[0:2] = (
        omega
        * (
            W_xi_ref_xy[0]
            -
            c_xy
        )
    )

    for k in range(N):
        r_xy = (
            W_support_ref[
                k,
                0:2,
            ]
        )

        xi_k = (
            W_xi_ref_xy[k]
        )

        c_xy_next = (
            r_xy
            +
            beta
            * (
                c_xy
                -
                r_xy
            )
            +
            sinh_term
            * (
                xi_k
                -
                r_xy
            )
        )

        v_xy_next = (
            omega
            * (
                W_xi_ref_xy[k + 1]
                -
                c_xy_next
            )
        )

        W_p_CoM_ref[k, 0:2] = (
            c_xy_next
        )

        W_p_CoM_ref[k, 2] = (
            com_height
        )

        W_v_CoM_ref[k, 0:2] = (
            v_xy_next
        )

        # Constant-height LIPM.
        W_v_CoM_ref[k, 2] = 0.0

        c_xy = (
            c_xy_next
        )

    return {
        "W_p_CoM_ref": W_p_CoM_ref,
        "W_v_CoM_ref": W_v_CoM_ref,
        "W_v_CoM_nominal_current": (
            W_v_CoM_nominal_current
        ),
        "beta_com": beta,
        "sinh_com": sinh_term,
    }


# ============================================================
# MAIN API
# ============================================================

def build_mpc_translational_reference(
    *,
    phase_preview: dict,
    W_p_L6_plan,
    W_p_R6_plan,
    W_p_CoM_nominal_current,
    com_height: float,
    gravity: float,
    horizon: int,
    dt_mpc: float,
    terminal_velocity_xy=None,
):
    """
    Build nominal translational references for SRBD MPC.

    Architecture
    ------------
        FSM phase/cursor
              |
              v
        phase_preview
              |
              v
        realized/current + future foot plan
              |
              v
        support/VRP horizon
              |
              v
        nominal DCM backward recursion
              |
              v
        nominal CoM reference horizon
              |
              v
        SRBD MPC

    IMPORTANT
    ---------
    There is NO measured-CoM correction in this module.

    Measured robot state x_k belongs to the MPC initial condition.
    The reference generator remains a nominal planner.

    W_p_CoM_nominal_current is the current NOMINAL reference state used
    only because DCM alone is insufficient to uniquely reconstruct CoM.
    """
    horizon = int(
        horizon
    )

    if horizon <= 0:
        raise ValueError(
            "horizon phải > 0."
        )

    com_height = _positive_scalar(
        com_height,
        "com_height",
    )

    gravity = _positive_scalar(
        gravity,
        "gravity",
    )

    dt_mpc = _positive_scalar(
        dt_mpc,
        "dt_mpc",
    )

    W_p_CoM_nominal_current = _as_vector3(
        W_p_CoM_nominal_current,
        "W_p_CoM_nominal_current",
    )

    omega = float(
        np.sqrt(
            gravity
            /
            com_height
        )
    )

    support = (
        build_support_reference_from_phase_preview(
            phase_preview=phase_preview,
            W_p_L6_plan=W_p_L6_plan,
            W_p_R6_plan=W_p_R6_plan,
            horizon=horizon,
        )
    )

    dcm = (
        compute_nominal_dcm_horizon(
            W_support_ref=(
                support[
                    "W_support_ref"
                ]
            ),
            omega=omega,
            dt_mpc=dt_mpc,
            terminal_velocity_xy=(
                terminal_velocity_xy
            ),
        )
    )

    com = (
        reconstruct_nominal_com_reference(
            W_xi_ref_xy=(
                dcm[
                    "W_xi_ref_xy"
                ]
            ),
            W_support_ref=(
                support[
                    "W_support_ref"
                ]
            ),
            W_p_CoM_nominal_current=(
                W_p_CoM_nominal_current
            ),
            com_height=com_height,
            omega=omega,
            dt_mpc=dt_mpc,
        )
    )

    return {
        # Main SRBD MPC reference.
        "W_p_CoM_ref": (
            com[
                "W_p_CoM_ref"
            ].copy()
        ),
        "W_v_CoM_ref": (
            com[
                "W_v_CoM_ref"
            ].copy()
        ),

        # Current nominal reference state.
        "W_p_CoM_nominal_current": (
            W_p_CoM_nominal_current.copy()
        ),
        "W_v_CoM_nominal_current": (
            com[
                "W_v_CoM_nominal_current"
            ].copy()
        ),

        # Support / DCM diagnostics.
        "W_support_ref": (
            support[
                "W_support_ref"
            ].copy()
        ),
        "W_xi_ref_xy": (
            dcm[
                "W_xi_ref_xy"
            ].copy()
        ),

        # Shared phase/contact diagnostics.
        "phase_sequence": list(
            support[
                "phase_sequence"
            ]
        ),
        "phase_progress_sequence": (
            support[
                "phase_progress_sequence"
            ].copy()
        ),
        "contact_schedule": (
            support[
                "contact_schedule"
            ].copy()
        ),

        "omega": omega,
        "dt_mpc": dt_mpc,
        "horizon": horizon,
    }
