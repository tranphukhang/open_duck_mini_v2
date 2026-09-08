import numpy as np


# ============================================================
# QUINTIC BLENDING
# ============================================================

def quintic_blend(tau):
    """
    Quintic interpolation:

        s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5

    with:

        s(0) = 0
        s(1) = 1

        ds/dtau(0) = 0
        ds/dtau(1) = 0

        d2s/dtau2(0) = 0
        d2s/dtau2(1) = 0
    """

    tau = np.clip(
        float(tau),
        0.0,
        1.0,
    )

    return (
        10.0 * tau**3
        - 15.0 * tau**4
        + 6.0 * tau**5
    )


def quintic_blend_derivative(tau):
    """
    Derivative ds/dtau.
    """

    tau = np.clip(
        float(tau),
        0.0,
        1.0,
    )

    return (
        30.0 * tau**2
        - 60.0 * tau**3
        + 30.0 * tau**4
    )


# ============================================================
# 1D QUINTIC INTERPOLATION
# ============================================================

def quintic_interpolate(
    start,
    target,
    time,
    duration,
):
    """
    Quintic interpolation from start to target.

    Returns:
        position
        velocity
    """

    if duration <= 0.0:

        raise ValueError(
            "duration must be positive."
        )

    t = np.clip(
        float(time),
        0.0,
        duration,
    )

    tau = (
        t / duration
    )

    s = quintic_blend(
        tau
    )

    ds_dtau = (
        quintic_blend_derivative(
            tau
        )
    )

    position = (
        start
        +
        s
        * (
            target
            -
            start
        )
    )

    velocity = (
        (
            target
            -
            start
        )
        *
        ds_dtau
        /
        duration
    )

    return (
        position,
        velocity
    )


# ============================================================
# SWING FOOT TRAJECTORY
# ============================================================

def compute_swing_trajectory(
    start_position,
    target_position,
    phase_time,
    duration,
    swing_height,
):
    """
    Generate swing-foot position and velocity references.

    Parameters
    ----------
    start_position : array-like, shape (3,)
        Swing foot position at lift-off.

    target_position : array-like, shape (3,)
        Desired landing position.

    phase_time : float
        Current time inside SINGLE_SUPPORT [s].

    duration : float
        Single-support duration [s].

    swing_height : float
        Additional vertical height at the apex [m].

    Returns
    -------
    p_ref : ndarray, shape (3,)
        Desired swing-foot position.

    v_ref : ndarray, shape (3,)
        Desired swing-foot linear velocity.

    Notes
    -----
    X and Y:
        one quintic trajectory over the whole SS phase.

    Z:
        two quintic trajectories:

            start -> apex
            apex  -> target

        where:

            apex_z =
                max(start_z, target_z)
                + swing_height

        Therefore the apex occurs exactly at:

            t = duration / 2
    """

    # ========================================================
    # CHECK INPUTS
    # ========================================================

    p0 = np.asarray(
        start_position,
        dtype=float,
    ).copy()

    pf = np.asarray(
        target_position,
        dtype=float,
    ).copy()

    if p0.shape != (3,):

        raise ValueError(
            "start_position must have shape (3,)."
        )

    if pf.shape != (3,):

        raise ValueError(
            "target_position must have shape (3,)."
        )

    if duration <= 0.0:

        raise ValueError(
            "duration must be positive."
        )

    if swing_height < 0.0:

        raise ValueError(
            "swing_height cannot be negative."
        )

    t = np.clip(
        float(phase_time),
        0.0,
        duration,
    )

    # ========================================================
    # X-Y TRAJECTORY
    # ========================================================

    x_ref, vx_ref = (
        quintic_interpolate(
            start=p0[0],
            target=pf[0],
            time=t,
            duration=duration,
        )
    )

    y_ref, vy_ref = (
        quintic_interpolate(
            start=p0[1],
            target=pf[1],
            time=t,
            duration=duration,
        )
    )

    # ========================================================
    # Z TRAJECTORY
    # ========================================================

    half_duration = (
        0.5 * duration
    )

    apex_z = (
        max(
            p0[2],
            pf[2],
        )
        +
        swing_height
    )

    # --------------------------------------------------------
    # First half:
    #
    # start -> apex
    # --------------------------------------------------------

    if t <= half_duration:

        z_ref, vz_ref = (
            quintic_interpolate(
                start=p0[2],
                target=apex_z,
                time=t,
                duration=half_duration,
            )
        )

    # --------------------------------------------------------
    # Second half:
    #
    # apex -> target
    # --------------------------------------------------------

    else:

        z_ref, vz_ref = (
            quintic_interpolate(
                start=apex_z,
                target=pf[2],
                time=(
                    t
                    -
                    half_duration
                ),
                duration=half_duration,
            )
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    p_ref = np.array(
        [
            x_ref,
            y_ref,
            z_ref,
        ],
        dtype=float,
    )

    v_ref = np.array(
        [
            vx_ref,
            vy_ref,
            vz_ref,
        ],
        dtype=float,
    )

    return (
        p_ref,
        v_ref,
    )