import numpy as np


# ============================================================
# SINUSOIDAL Z STANDING REFERENCE
# ============================================================

def build_sinusoidal_z_reference(
    control_time,
    x_initial,
    horizon,
    dt_mpc,
    amplitude_z=0.005,
    frequency_z=0.2,
    phase_z=0.0,
):
    """
    Build a standing SRBD reference with sinusoidal CoM motion
    only along the World z axis.

    Reference
    ---------

        x_ref = x_initial
        y_ref = y_initial

        z_ref(t)
            =
        z0 + A_z sin(2*pi*f_z*t + phase_z)

        vz_ref(t)
            =
        2*pi*f_z*A_z cos(2*pi*f_z*t + phase_z)

        euler_ref
            =
        euler_initial

        vx_ref = 0
        vy_ref = 0

        omega_ref = 0

    MPC horizon convention
    ----------------------

    At current MPC update time t_k:

        X_ref =
        [
            x_ref(t_k + 1*dt_mpc)
            x_ref(t_k + 2*dt_mpc)
            ...
            x_ref(t_k + N*dt_mpc)
        ]

    Therefore the reference is aligned with the predicted states:

        x[k+1|k], ..., x[k+N|k]

    Returns
    -------

    x_ref_current : np.ndarray, shape (13,)

        Reference evaluated at the CURRENT control time t_k.
        This is mainly useful for diagnostics / plotting.

    X_ref : np.ndarray, shape (13*N,)

        Stacked MPC reference over the prediction horizon.
    """

    control_time = float(
        control_time
    )

    x_initial = np.asarray(
        x_initial,
        dtype=float,
    ).reshape(-1)

    horizon = int(
        horizon
    )

    dt_mpc = float(
        dt_mpc
    )

    amplitude_z = float(
        amplitude_z
    )

    frequency_z = float(
        frequency_z
    )

    phase_z = float(
        phase_z
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    if x_initial.shape != (13,):
        raise ValueError(
            "x_initial phải có shape (13,)."
        )

    if not np.all(
        np.isfinite(
            x_initial
        )
    ):
        raise ValueError(
            "x_initial chứa NaN hoặc Inf."
        )

    if not np.isfinite(
        control_time
    ):
        raise ValueError(
            "control_time không finite."
        )

    if horizon <= 0:
        raise ValueError(
            "horizon phải > 0."
        )

    if (
        not np.isfinite(
            dt_mpc
        )
        or
        dt_mpc <= 0.0
    ):
        raise ValueError(
            "dt_mpc phải hữu hạn và > 0."
        )

    if (
        not np.isfinite(
            amplitude_z
        )
        or
        amplitude_z < 0.0
    ):
        raise ValueError(
            "amplitude_z phải hữu hạn và >= 0."
        )

    if (
        not np.isfinite(
            frequency_z
        )
        or
        frequency_z < 0.0
    ):
        raise ValueError(
            "frequency_z phải hữu hạn và >= 0."
        )

    if not np.isfinite(
        phase_z
    ):
        raise ValueError(
            "phase_z không finite."
        )

    # ========================================================
    # CONSTANT REFERENCE COMPONENTS
    # ========================================================

    x0 = float(
        x_initial[0]
    )

    y0 = float(
        x_initial[1]
    )

    z0 = float(
        x_initial[2]
    )

    euler0 = np.asarray(
        x_initial[3:6],
        dtype=float,
    ).copy()

    omega_z = (
        2.0
        *
        np.pi
        *
        frequency_z
    )

    # ========================================================
    # HELPER: BUILD ONE REFERENCE STATE
    # ========================================================

    def _build_one_reference(
        t_ref
    ):
        angle = (
            omega_z
            *
            t_ref
            +
            phase_z
        )

        z_ref = (
            z0
            +
            amplitude_z
            *
            np.sin(
                angle
            )
        )

        vz_ref = (
            omega_z
            *
            amplitude_z
            *
            np.cos(
                angle
            )
        )

        x_ref = np.zeros(
            13,
            dtype=float,
        )

        # Position.
        x_ref[0] = x0
        x_ref[1] = y0
        x_ref[2] = z_ref

        # Orientation.
        x_ref[3:6] = euler0

        # Linear velocity.
        x_ref[6] = 0.0
        x_ref[7] = 0.0
        x_ref[8] = vz_ref

        # Angular velocity.
        x_ref[9:12] = 0.0

        # Homogeneous constant state.
        x_ref[12] = 1.0

        return x_ref

    # ========================================================
    # CURRENT REFERENCE
    # ========================================================

    x_ref_current = _build_one_reference(
        control_time
    )

    # ========================================================
    # FUTURE MPC REFERENCE
    # ========================================================

    X_ref_matrix = np.zeros(
        (
            horizon,
            13,
        ),
        dtype=float,
    )

    for j in range(
        horizon
    ):
        t_ref = (
            control_time
            +
            (j + 1)
            *
            dt_mpc
        )

        X_ref_matrix[j] = (
            _build_one_reference(
                t_ref
            )
        )

    X_ref = X_ref_matrix.reshape(-1)

    expected_shape = (
        13
        *
        horizon,
    )

    if X_ref.shape != expected_shape:
        raise RuntimeError(
            f"X_ref phải có shape {expected_shape}, "
            f"nhưng nhận được {X_ref.shape}."
        )

    if not np.all(
        np.isfinite(
            X_ref
        )
    ):
        raise RuntimeError(
            "X_ref chứa NaN hoặc Inf."
        )

    return (
        x_ref_current,
        X_ref,
    )
