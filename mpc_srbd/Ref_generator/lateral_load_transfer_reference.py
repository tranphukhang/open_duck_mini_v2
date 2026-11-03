import numpy as np


# ============================================================
# SMOOTHSTEP
#
# h(alpha) = 3 alpha^2 - 2 alpha^3
#
# h(0) = 0
# h(1) = 1
#
# dh/dt = 0 at both endpoints.
# ============================================================

def _smoothstep01(alpha):

    alpha = float(
        np.clip(
            alpha,
            0.0,
            1.0,
        )
    )

    h = (
        3.0 * alpha**2
        -
        2.0 * alpha**3
    )

    dh_dalpha = (
        6.0 * alpha
        -
        6.0 * alpha**2
    )

    return (
        float(h),
        float(dh_dalpha),
    )


# ============================================================
# SMOOTH TRANSITION
# ============================================================

def _transition(
    tau,
    duration,
    y0,
    y1,
):

    tau = float(tau)

    duration = float(duration)

    y0 = float(y0)

    y1 = float(y1)


    if duration <= 0.0:

        return (
            y1,
            0.0,
        )


    alpha = (
        tau
        /
        duration
    )


    h, dh_dalpha = (
        _smoothstep01(
            alpha
        )
    )


    y = (
        y0
        +
        (
            y1
            -
            y0
        )
        *
        h
    )


    vy = (
        (
            y1
            -
            y0
        )
        *
        dh_dalpha
        /
        duration
    )


    return (
        float(y),
        float(vy),
    )


# ============================================================
# NORMAL GAIT CYCLE
#
# first swing leg = R:
#
# SSP_L
#   ->
# DSP1
#   ->
# SSP_R
#   ->
# DSP2
#
#
# Lateral reference:
#
# SSP_L:
#     +A
#
# DSP1:
#     +A -> -A
#
# SSP_R:
#     -A
#
# DSP2:
#     -A -> +A
# ============================================================

def _cycle_right_swing_first(
    local_time,
    T_ssp_L,
    T_dsp1,
    T_ssp_R,
    T_dsp2,
    amplitude,
):

    period = (
        T_ssp_L
        +
        T_dsp1
        +
        T_ssp_R
        +
        T_dsp2
    )


    if period <= 0.0:

        return (
            0.0,
            0.0,
        )


    s = (
        float(local_time)
        %
        period
    )


    A = float(
        amplitude
    )


    # ========================================================
    # SSP_L
    #
    # Left stance:
    #
    # y_ref = +A
    # ========================================================

    if s < T_ssp_L:

        return (
            +A,
            0.0,
        )


    s -= T_ssp_L


    # ========================================================
    # DSP1
    #
    # Transfer:
    #
    # Left stance
    #     ->
    # Right stance
    #
    # +A -> -A
    # ========================================================

    if s < T_dsp1:

        return (
            _transition(
                tau=s,
                duration=T_dsp1,
                y0=+A,
                y1=-A,
            )
        )


    s -= T_dsp1


    # ========================================================
    # SSP_R
    #
    # Right stance:
    #
    # y_ref = -A
    # ========================================================

    if s < T_ssp_R:

        return (
            -A,
            0.0,
        )


    s -= T_ssp_R


    # ========================================================
    # DSP2
    #
    # Transfer:
    #
    # Right stance
    #     ->
    # Left stance
    #
    # -A -> +A
    # ========================================================

    return (
        _transition(
            tau=s,
            duration=T_dsp2,
            y0=-A,
            y1=+A,
        )
    )


# ============================================================
# NORMAL GAIT CYCLE
#
# first swing leg = L
#
# SSP_R
#   ->
# DSP2
#   ->
# SSP_L
#   ->
# DSP1
# ============================================================

def _cycle_left_swing_first(
    local_time,
    T_ssp_L,
    T_dsp1,
    T_ssp_R,
    T_dsp2,
    amplitude,
):

    period = (
        T_ssp_R
        +
        T_dsp2
        +
        T_ssp_L
        +
        T_dsp1
    )


    if period <= 0.0:

        return (
            0.0,
            0.0,
        )


    s = (
        float(local_time)
        %
        period
    )


    A = float(
        amplitude
    )


    # ========================================================
    # SSP_R
    # ========================================================

    if s < T_ssp_R:

        return (
            -A,
            0.0,
        )


    s -= T_ssp_R


    # ========================================================
    # DSP2
    #
    # -A -> +A
    # ========================================================

    if s < T_dsp2:

        return (
            _transition(
                tau=s,
                duration=T_dsp2,
                y0=-A,
                y1=+A,
            )
        )


    s -= T_dsp2


    # ========================================================
    # SSP_L
    # ========================================================

    if s < T_ssp_L:

        return (
            +A,
            0.0,
        )


    s -= T_ssp_L


    # ========================================================
    # DSP1
    #
    # +A -> -A
    # ========================================================

    return (
        _transition(
            tau=s,
            duration=T_dsp1,
            y0=+A,
            y1=-A,
        )
    )


# ============================================================
# ONE TIME SAMPLE
# ============================================================

def lateral_load_transfer_at_time(
    t,
    startup_hold_time,
    dt_mpc,
    ssp_durations,
    dsp_durations,
    amplitude=0.04,
    first_swing_leg="R",
):
    """
    Return:

        y_offset
        vy_offset

    at absolute simulation time t.

    Convention
    ----------

    +y:
        shift CoM reference toward LEFT foot.

    -y:
        shift CoM reference toward RIGHT foot.
    """

    t = float(
        t
    )

    startup_hold_time = float(
        startup_hold_time
    )

    dt_mpc = float(
        dt_mpc
    )

    amplitude = float(
        amplitude
    )


    ssp_durations = np.asarray(
        ssp_durations,
        dtype=float,
    ).reshape(
        2
    )


    dsp_durations = np.asarray(
        dsp_durations,
        dtype=float,
    ).reshape(
        2
    )


    first_swing_leg = (
        str(
            first_swing_leg
        )
        .strip()
        .upper()
    )


    if dt_mpc <= 0.0:

        raise ValueError(
            "dt_mpc must be > 0."
        )


    if startup_hold_time < 0.0:

        raise ValueError(
            "startup_hold_time must be >= 0."
        )


    if amplitude < 0.0:

        raise ValueError(
            "amplitude must be >= 0."
        )


    if first_swing_leg not in (
        "L",
        "R",
    ):

        raise ValueError(
            "first_swing_leg must be 'L' or 'R'."
        )


    # ========================================================
    # PHYSICAL PHASE DURATIONS
    # ========================================================

    T_ssp_L = (
        ssp_durations[0]
        *
        dt_mpc
    )


    T_ssp_R = (
        ssp_durations[1]
        *
        dt_mpc
    )


    T_dsp1 = (
        dsp_durations[0]
        *
        dt_mpc
    )


    T_dsp2 = (
        dsp_durations[1]
        *
        dt_mpc
    )


    # ========================================================
    # STARTUP HOLD
    #
    # Robot remains centered.
    # ========================================================

    if t < startup_hold_time:

        return (
            0.0,
            0.0,
        )


    tau = (
        t
        -
        startup_hold_time
    )


    A = amplitude


    # ========================================================
    # RIGHT FOOT SWINGS FIRST
    #
    # Startup:
    #
    # centered DS
    #   ->
    # DSP2
    #   ->
    # SSP_L
    #
    # Therefore first transfer:
    #
    # 0 -> +A
    # ========================================================

    if first_swing_leg == "R":

        if tau < T_dsp2:

            return (
                _transition(
                    tau=tau,
                    duration=T_dsp2,
                    y0=0.0,
                    y1=+A,
                )
            )


        tau -= T_dsp2


        return (
            _cycle_right_swing_first(
                local_time=tau,
                T_ssp_L=T_ssp_L,
                T_dsp1=T_dsp1,
                T_ssp_R=T_ssp_R,
                T_dsp2=T_dsp2,
                amplitude=A,
            )
        )


    # ========================================================
    # LEFT FOOT SWINGS FIRST
    #
    # centered DS
    #   ->
    # DSP1
    #   ->
    # SSP_R
    #
    # Therefore:
    #
    # 0 -> -A
    # ========================================================

    if tau < T_dsp1:

        return (
            _transition(
                tau=tau,
                duration=T_dsp1,
                y0=0.0,
                y1=-A,
            )
        )


    tau -= T_dsp1


    return (
        _cycle_left_swing_first(
            local_time=tau,
            T_ssp_L=T_ssp_L,
            T_dsp1=T_dsp1,
            T_ssp_R=T_ssp_R,
            T_dsp2=T_dsp2,
            amplitude=A,
        )
    )


# ============================================================
# BUILD COMPLETE REFERENCE PROFILE
# ============================================================

def build_lateral_load_transfer_profile(
    t_now,
    horizon_length,
    dt_mpc,
    startup_hold_time,
    ssp_durations,
    dsp_durations,
    amplitude=0.04,
    first_swing_leg="R",
):
    """
    Build lateral load-transfer reference over the full
    reference horizon.

    Returns
    -------

    dict:

        y_offset
            shape (horizon_length,)

        vy_offset
            shape (horizon_length,)

        time
            absolute time of each knot
    """

    horizon_length = int(
        horizon_length
    )


    if horizon_length <= 0:

        raise ValueError(
            "horizon_length must be > 0."
        )


    y_offset = np.zeros(
        horizon_length,
        dtype=float,
    )


    vy_offset = np.zeros(
        horizon_length,
        dtype=float,
    )


    time = np.zeros(
        horizon_length,
        dtype=float,
    )


    for i in range(
        horizon_length
    ):

        t_i = (
            float(
                t_now
            )
            +
            float(
                i
            )
            *
            float(
                dt_mpc
            )
        )


        y_i, vy_i = (
            lateral_load_transfer_at_time(
                t=t_i,
                startup_hold_time=(
                    startup_hold_time
                ),
                dt_mpc=dt_mpc,
                ssp_durations=(
                    ssp_durations
                ),
                dsp_durations=(
                    dsp_durations
                ),
                amplitude=amplitude,
                first_swing_leg=(
                    first_swing_leg
                ),
            )
        )


        time[i] = t_i

        y_offset[i] = y_i

        vy_offset[i] = vy_i


    return {
        "time":
            time,

        "y_offset":
            y_offset,

        "vy_offset":
            vy_offset,
    }