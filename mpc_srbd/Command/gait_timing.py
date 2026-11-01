import numpy as np


# ============================================================
# BUILD GAIT TIMING
# ============================================================

def build_gait_timing(
    T_step,
    horizon=10,
    N_SS=8,
    N_DS=2
):
    """
    Build the nominal gait timing used by the MPC prediction.

    Design principle
    ----------------

    One nominal walking step is defined as:

        T_step = T_SS + T_DS

    The MPC prediction horizon contains exactly:

        N = horizon

    prediction intervals over one nominal step.

    Therefore:

        dt_mpc = T_step / N


    The prediction intervals are divided into:

        N_SS = number of Single Support intervals

        N_DS = number of Double Support intervals

    with:

        N_SS + N_DS = N


    Hence:

        T_SS = N_SS * dt_mpc

        T_DS = N_DS * dt_mpc


    Parameters
    ----------
    T_step : float

        Nominal duration of one walking step [s].

        One step consists of:

            Single Support
                +
            Double Support


    horizon : int, optional

        MPC prediction horizon N.

        Default:
            10


    N_SS : int, optional

        Number of prediction intervals assigned to
        Single Support.

        Default:
            8


    N_DS : int, optional

        Number of prediction intervals assigned to
        Double Support.

        Default:
            2


    Returns
    -------
    timing : dict

        Dictionary containing:

            timing["T_step"]

            timing["horizon"]

            timing["dt_mpc"]

            timing["N_SS"]

            timing["N_DS"]

            timing["T_SS"]

            timing["T_DS"]

            timing["SS_ratio"]

            timing["DS_ratio"]


    Example
    -------

    For:

        T_step = 0.5 s

        horizon = 10

        N_SS = 8

        N_DS = 2

    we obtain:

        dt_mpc = 0.05 s

        T_SS = 0.40 s

        T_DS = 0.10 s
    """


    # ========================================================
    # 1. INPUT CONVERSION
    # ========================================================

    T_step = float(
        T_step
    )


    horizon = int(
        horizon
    )


    N_SS = int(
        N_SS
    )


    N_DS = int(
        N_DS
    )


    # ========================================================
    # 2. INPUT VALIDATION
    # ========================================================

    if not np.isfinite(
        T_step
    ):

        raise ValueError(
            "T_step chứa NaN hoặc Inf."
        )


    if T_step <= 0.0:

        raise ValueError(
            "T_step phải lớn hơn 0."
        )


    if horizon <= 0:

        raise ValueError(
            "horizon phải lớn hơn 0."
        )


    if N_SS <= 0:

        raise ValueError(
            "N_SS phải lớn hơn 0."
        )


    if N_DS <= 0:

        raise ValueError(
            "N_DS phải lớn hơn 0."
        )


    # --------------------------------------------------------
    # Number of SS and DS intervals must exactly fill
    # one MPC horizon.
    # --------------------------------------------------------

    if (
        N_SS
        +
        N_DS
        !=
        horizon
    ):

        raise ValueError(
            "N_SS + N_DS phải bằng horizon. "
            f"Nhận được: "
            f"N_SS={N_SS}, "
            f"N_DS={N_DS}, "
            f"horizon={horizon}."
        )


    # ========================================================
    # 3. MPC PREDICTION TIME STEP
    #
    #     dt_mpc = T_step / N
    # ========================================================

    dt_mpc = (
        T_step
        /
        horizon
    )


    # ========================================================
    # 4. SINGLE SUPPORT DURATION
    #
    #     T_SS = N_SS * dt_mpc
    # ========================================================

    T_SS = (
        N_SS
        *
        dt_mpc
    )


    # ========================================================
    # 5. DOUBLE SUPPORT DURATION
    #
    #     T_DS = N_DS * dt_mpc
    # ========================================================

    T_DS = (
        N_DS
        *
        dt_mpc
    )


    # ========================================================
    # 6. SUPPORT-PHASE RATIOS
    # ========================================================

    SS_ratio = (
        N_SS
        /
        horizon
    )


    DS_ratio = (
        N_DS
        /
        horizon
    )


    # ========================================================
    # 7. INTERNAL CONSISTENCY CHECK
    # ========================================================

    reconstructed_T_step = (
        T_SS
        +
        T_DS
    )


    if not np.isclose(
        reconstructed_T_step,
        T_step,
        rtol=1e-12,
        atol=1e-12
    ):

        raise RuntimeError(
            "Timing inconsistency: "
            "T_SS + T_DS != T_step."
        )


    reconstructed_horizon_time = (
        horizon
        *
        dt_mpc
    )


    if not np.isclose(
        reconstructed_horizon_time,
        T_step,
        rtol=1e-12,
        atol=1e-12
    ):

        raise RuntimeError(
            "Timing inconsistency: "
            "horizon * dt_mpc != T_step."
        )


    # ========================================================
    # 8. OUTPUT
    # ========================================================

    timing = {
        "T_step": T_step,

        "horizon": horizon,

        "dt_mpc": dt_mpc,

        "N_SS": N_SS,

        "N_DS": N_DS,

        "T_SS": T_SS,

        "T_DS": T_DS,

        "SS_ratio": SS_ratio,

        "DS_ratio": DS_ratio,
    }


    return timing


# ============================================================
# GET NOMINAL PHASE DURATION
# ============================================================

def get_nominal_phase_duration(
    phase,
    timing
):
    """
    Return the nominal duration of a gait phase.

    Supported phases
    ----------------

        "LSS"

        "RSS"

        "DS_L_TO_R"

        "DS_R_TO_L"


    Parameters
    ----------
    phase : str

        Current gait phase.


    timing : dict

        Output of:

            build_gait_timing(...)


    Returns
    -------
    duration : float

        Nominal phase duration [s].
    """


    phase = str(
        phase
    ).upper()


    if phase in (
        "LSS",
        "RSS"
    ):

        return float(
            timing["T_SS"]
        )


    if phase in (
        "DS_L_TO_R",
        "DS_R_TO_L"
    ):

        return float(
            timing["T_DS"]
        )


    raise ValueError(
        "Unknown gait phase: "
        f"{phase}"
    )