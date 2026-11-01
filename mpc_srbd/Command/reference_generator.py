import numpy as np


# ============================================================
# GAIT
# ============================================================

try:

    from .gait_timing import (
        get_nominal_phase_duration
    )

    from .contact_schedule import (
        normalize_gait_phase,
        get_next_gait_phase
    )

except ImportError:

    from gait_timing import (
        get_nominal_phase_duration
    )

    from contact_schedule import (
        normalize_gait_phase,
        get_next_gait_phase
    )


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_GRAVITY = 9.81


# ============================================================
# VALIDATION
# ============================================================

def _validate_vector(
    value,
    name,
    size
):
    value = np.asarray(
        value,
        dtype=float
    ).reshape(-1)


    if value.shape != (
        size,
    ):

        raise ValueError(
            f"{name} phải có shape ({size},), "
            f"nhưng nhận được {value.shape}."
        )


    if not np.all(
        np.isfinite(
            value
        )
    ):

        raise ValueError(
            f"{name} chứa NaN hoặc Inf."
        )


    return value


def _validate_support_points(
    support_points
):
    """
    Validate ordered future single-support footholds.

    Convention
    ----------

    support_points[0]
        =
    current / outgoing support foot.


    support_points[1]
        =
    next support foot.


    support_points[2]
        =
    support foot after that.


    support_points[3]
        =
    next one after that.


    Examples
    --------

    Current LSS:

        [
            current left support,
            next right foothold,
            next left foothold,
            next right foothold
        ]


    Current DS_L_TO_R:

        [
            outgoing left support,
            incoming right support,
            next left foothold,
            next right foothold
        ]


    Current RSS:

        [
            current right support,
            next left foothold,
            next right foothold,
            next left foothold
        ]


    Current DS_R_TO_L:

        [
            outgoing right support,
            incoming left support,
            next right foothold,
            next left foothold
        ]


    Input may be:

        shape (M, 2)

    or:

        shape (M, 3)

    Only x-y components are used by the flat-ground DCM model.
    """


    support_points = np.asarray(
        support_points,
        dtype=float
    )


    if support_points.ndim != 2:

        raise ValueError(
            "support_points phải là ma trận 2D."
        )


    if support_points.shape[1] not in (
        2,
        3
    ):

        raise ValueError(
            "support_points phải có shape (M,2) "
            "hoặc (M,3)."
        )


    if support_points.shape[0] < 3:

        raise ValueError(
            "Cần ít nhất 3 support points."
        )


    if not np.all(
        np.isfinite(
            support_points
        )
    ):

        raise ValueError(
            "support_points chứa NaN hoặc Inf."
        )


    return (
        support_points[
            :,
            0:2
        ].copy()
    )


# ============================================================
# LIPM / DCM PARAMETERS
# ============================================================

def compute_omega0(
    com_height,
    gravity=DEFAULT_GRAVITY
):
    """
    Flat-ground LIPM natural frequency:

        omega_0 = sqrt(g / z_0)
    """


    com_height = float(
        com_height
    )


    gravity = float(
        gravity
    )


    if (
        not np.isfinite(
            com_height
        )
        or
        com_height <= 0.0
    ):

        raise ValueError(
            "com_height phải hữu hạn và > 0."
        )


    if (
        not np.isfinite(
            gravity
        )
        or
        gravity <= 0.0
    ):

        raise ValueError(
            "gravity phải hữu hạn và > 0."
        )


    return float(
        np.sqrt(
            gravity
            /
            com_height
        )
    )


# ============================================================
# DCM FROM CoM STATE
# ============================================================

def compute_dcm_xy(
    W_p_CoM,
    W_v_CoM,
    omega0
):
    """
    Flat-ground DCM:

        xi
            =
        p_CoM,xy
            +
        v_CoM,xy / omega0
    """


    W_p_CoM = _validate_vector(
        W_p_CoM,
        "W_p_CoM",
        3
    )


    W_v_CoM = _validate_vector(
        W_v_CoM,
        "W_v_CoM",
        3
    )


    omega0 = float(
        omega0
    )


    if (
        not np.isfinite(
            omega0
        )
        or
        omega0 <= 0.0
    ):

        raise ValueError(
            "omega0 phải hữu hạn và > 0."
        )


    return (

        W_p_CoM[
            0:2
        ]

        +

        W_v_CoM[
            0:2
        ]
        /
        omega0
    )


# ============================================================
# DESIRED INITIAL DCM FOR A FUTURE SS PHASE
#
# If:
#
#   xi_ini
#       =
#   p_support
#       +
#   exp(-omega*T_SS)
#   (p_next - p_support)
#
# then after one complete SS:
#
#   xi(T_SS) = p_next
# ============================================================

def compute_ss_initial_dcm(
    W_p_support_xy,
    W_p_next_xy,
    T_SS,
    omega0
):
    W_p_support_xy = _validate_vector(
        W_p_support_xy,
        "W_p_support_xy",
        2
    )


    W_p_next_xy = _validate_vector(
        W_p_next_xy,
        "W_p_next_xy",
        2
    )


    T_SS = float(
        T_SS
    )


    omega0 = float(
        omega0
    )


    if T_SS <= 0.0:

        raise ValueError(
            "T_SS phải > 0."
        )


    return (

        W_p_support_xy

        +

        np.exp(
            -omega0
            *
            T_SS
        )

        *

        (
            W_p_next_xy
            -
            W_p_support_xy
        )
    )


# ============================================================
# SINGLE SUPPORT DCM
#
# xi(t)
#   =
# p_sup
#   +
# exp(omega t)
# (xi_0 - p_sup)
#
#
# xi_dot(t)
#   =
# omega (xi(t) - p_sup)
# ============================================================

def evaluate_ss_dcm(
    xi_start,
    W_p_support_xy,
    local_time,
    omega0
):
    xi_start = _validate_vector(
        xi_start,
        "xi_start",
        2
    )


    W_p_support_xy = _validate_vector(
        W_p_support_xy,
        "W_p_support_xy",
        2
    )


    local_time = float(
        local_time
    )


    if local_time < 0.0:

        raise ValueError(
            "local_time không được âm."
        )


    exponential = np.exp(
        omega0
        *
        local_time
    )


    xi = (

        W_p_support_xy

        +

        exponential

        *

        (
            xi_start
            -
            W_p_support_xy
        )
    )


    xi_dot = (

        omega0

        *

        (
            xi
            -
            W_p_support_xy
        )
    )


    return (
        xi,
        xi_dot
    )


# ============================================================
# DOUBLE SUPPORT DCM
#
# Cubic Hermite interpolation:
#
#   xi(0) = xi_start
#   xi_dot(0) = xi_dot_start
#
#   xi(T) = xi_end
#   xi_dot(T) = xi_dot_end
#
# Therefore both DCM position and DCM velocity remain
# continuous at SS <-> DS boundaries.
# ============================================================

def evaluate_ds_dcm(
    xi_start,
    xi_dot_start,
    xi_end,
    xi_dot_end,
    local_time,
    duration
):
    xi_start = _validate_vector(
        xi_start,
        "xi_start",
        2
    )


    xi_dot_start = _validate_vector(
        xi_dot_start,
        "xi_dot_start",
        2
    )


    xi_end = _validate_vector(
        xi_end,
        "xi_end",
        2
    )


    xi_dot_end = _validate_vector(
        xi_dot_end,
        "xi_dot_end",
        2
    )


    duration = float(
        duration
    )


    local_time = float(
        local_time
    )


    if duration <= 0.0:

        raise ValueError(
            "duration phải > 0."
        )


    if (
        local_time < 0.0
        or
        local_time > duration + 1e-12
    ):

        raise ValueError(
            "local_time phải nằm trong [0, duration]."
        )


    local_time = float(
        np.clip(
            local_time,
            0.0,
            duration
        )
    )


    s = (
        local_time
        /
        duration
    )


    # --------------------------------------------------------
    # Hermite basis
    # --------------------------------------------------------

    h00 = (
        2.0 * s ** 3
        -
        3.0 * s ** 2
        +
        1.0
    )


    h10 = (
        s ** 3
        -
        2.0 * s ** 2
        +
        s
    )


    h01 = (
        -2.0 * s ** 3
        +
        3.0 * s ** 2
    )


    h11 = (
        s ** 3
        -
        s ** 2
    )


    xi = (

        h00
        *
        xi_start

        +

        h10
        *
        duration
        *
        xi_dot_start

        +

        h01
        *
        xi_end

        +

        h11
        *
        duration
        *
        xi_dot_end
    )


    # --------------------------------------------------------
    # Derivative with respect to s
    # --------------------------------------------------------

    dh00_ds = (
        6.0 * s ** 2
        -
        6.0 * s
    )


    dh10_ds = (
        3.0 * s ** 2
        -
        4.0 * s
        +
        1.0
    )


    dh01_ds = (
        -6.0 * s ** 2
        +
        6.0 * s
    )


    dh11_ds = (
        3.0 * s ** 2
        -
        2.0 * s
    )


    dxi_ds = (

        dh00_ds
        *
        xi_start

        +

        dh10_ds
        *
        duration
        *
        xi_dot_start

        +

        dh01_ds
        *
        xi_end

        +

        dh11_ds
        *
        duration
        *
        xi_dot_end
    )


    xi_dot = (
        dxi_ds
        /
        duration
    )


    return (
        xi,
        xi_dot
    )


# ============================================================
# PHASE TYPE
# ============================================================

def _is_single_support(
    phase
):
    return phase in (
        "LSS",
        "RSS"
    )


def _is_double_support(
    phase
):
    return phase in (
        "DS_L_TO_R",
        "DS_R_TO_L"
    )


# ============================================================
# BUILD PIECEWISE DCM SEGMENTS
#
# support_index represents the outgoing/current SS support.
#
# Example:
#
# current LSS:
#
#   support_index = 0
#
# LSS:
#   support_points[0]
#
# DS:
#   support_points[0] -> support_points[1]
#
# RSS:
#   support_points[1]
#
# DS:
#   support_points[1] -> support_points[2]
# ============================================================

def build_dcm_segments(
    current_phase,
    t_phase,
    timing,
    xi_current,
    support_points,
    omega0,
    horizon=None,
    dt_mpc=None
):
    """
    Build continuous piecewise DCM trajectory segments.

    Current DCM is initialized from the measured CoM state.

    Actual current phase is supplied externally from the
    contact-synchronized FSM.

    Future phase evolution uses nominal timing.
    """


    current_phase = normalize_gait_phase(
        current_phase
    )


    t_phase = float(
        t_phase
    )


    if (
        not np.isfinite(
            t_phase
        )
        or
        t_phase < 0.0
    ):

        raise ValueError(
            "t_phase phải hữu hạn và >= 0."
        )


    xi_current = _validate_vector(
        xi_current,
        "xi_current",
        2
    )


    support_points = (
        _validate_support_points(
            support_points
        )
    )


    omega0 = float(
        omega0
    )


    if horizon is None:

        horizon = int(
            timing[
                "horizon"
            ]
        )

    else:

        horizon = int(
            horizon
        )


    if dt_mpc is None:

        dt_mpc = float(
            timing[
                "dt_mpc"
            ]
        )

    else:

        dt_mpc = float(
            dt_mpc
        )


    if horizon <= 0:

        raise ValueError(
            "horizon phải > 0."
        )


    if dt_mpc <= 0.0:

        raise ValueError(
            "dt_mpc phải > 0."
        )


    horizon_time = (
        horizon
        *
        dt_mpc
    )


    T_SS = float(
        timing[
            "T_SS"
        ]
    )


    # ========================================================
    # INITIAL STATE
    # ========================================================

    phase = (
        current_phase
    )


    support_index = 0


    xi_segment_start = (
        xi_current.copy()
    )


    total_time = 0.0


    first_segment = True


    segments = []


    # ========================================================
    # BUILD UNTIL COMPLETE MPC HORIZON IS COVERED
    # ========================================================

    while (
        total_time
        <
        horizon_time
        -
        1e-12
    ):

        nominal_duration = float(
            get_nominal_phase_duration(
                phase,
                timing
            )
        )


        # ----------------------------------------------------
        # Current phase may already be partially completed.
        # Future phases always start at t_phase = 0.
        # ----------------------------------------------------

        if first_segment:

            elapsed = min(
                t_phase,
                nominal_duration
            )


            duration = max(
                nominal_duration
                -
                elapsed,
                0.0
            )

        else:

            duration = (
                nominal_duration
            )


        # ====================================================
        # SINGLE SUPPORT
        # ====================================================

        if _is_single_support(
            phase
        ):

            if support_index >= (
                support_points.shape[0]
            ):

                raise ValueError(
                    "Không đủ support_points "
                    "để build DCM horizon."
                )


            W_p_support = (
                support_points[
                    support_index
                ]
            )


            if duration > 1e-12:

                (
                    xi_end,
                    xi_dot_end

                ) = evaluate_ss_dcm(

                    xi_start=(
                        xi_segment_start
                    ),

                    W_p_support_xy=(
                        W_p_support
                    ),

                    local_time=duration,

                    omega0=omega0
                )


                segments.append(
                    {
                        "kind":
                            "SS",

                        "phase":
                            phase,

                        "t_start":
                            total_time,

                        "t_end":
                            total_time
                            +
                            duration,

                        "duration":
                            duration,

                        "xi_start":
                            xi_segment_start.copy(),

                        "xi_end":
                            xi_end.copy(),

                        "support":
                            W_p_support.copy(),
                    }
                )


                xi_segment_start = (
                    xi_end.copy()
                )


                total_time += (
                    duration
                )


            # ------------------------------------------------
            # SS -> DS
            #
            # support_index does not change yet.
            # ------------------------------------------------

            phase = (
                get_next_gait_phase(
                    phase
                )
            )


        # ====================================================
        # DOUBLE SUPPORT
        # ====================================================

        elif _is_double_support(
            phase
        ):

            # ------------------------------------------------
            # Need:
            #
            # old support
            # new support
            # support after new support
            # ------------------------------------------------

            if (
                support_index
                +
                2
                >=
                support_points.shape[0]
            ):

                raise ValueError(
                    "Không đủ support_points để build "
                    "DS DCM transition. "
                    "Cần old support, new support "
                    "và support kế tiếp."
                )


            W_p_old = (
                support_points[
                    support_index
                ]
            )


            W_p_new = (
                support_points[
                    support_index
                    +
                    1
                ]
            )


            W_p_after_new = (
                support_points[
                    support_index
                    +
                    2
                ]
            )


            # ------------------------------------------------
            # Desired DCM at beginning of next SS.
            #
            # This guarantees that after one full next SS:
            #
            #   xi -> W_p_after_new
            # ------------------------------------------------

            xi_next_ss_initial = (
                compute_ss_initial_dcm(

                    W_p_support_xy=(
                        W_p_new
                    ),

                    W_p_next_xy=(
                        W_p_after_new
                    ),

                    T_SS=T_SS,

                    omega0=omega0
                )
            )


            # ------------------------------------------------
            # DS boundary velocities.
            # ------------------------------------------------

            xi_dot_start = (

                omega0

                *

                (
                    xi_segment_start
                    -
                    W_p_old
                )
            )


            xi_dot_end = (

                omega0

                *

                (
                    xi_next_ss_initial
                    -
                    W_p_new
                )
            )


            if duration > 1e-12:

                segments.append(
                    {
                        "kind":
                            "DS",

                        "phase":
                            phase,

                        "t_start":
                            total_time,

                        "t_end":
                            total_time
                            +
                            duration,

                        "duration":
                            duration,

                        "xi_start":
                            xi_segment_start.copy(),

                        "xi_dot_start":
                            xi_dot_start.copy(),

                        "xi_end":
                            xi_next_ss_initial.copy(),

                        "xi_dot_end":
                            xi_dot_end.copy(),

                        "old_support":
                            W_p_old.copy(),

                        "new_support":
                            W_p_new.copy(),
                    }
                )


                xi_segment_start = (
                    xi_next_ss_initial.copy()
                )


                total_time += (
                    duration
                )


            # ------------------------------------------------
            # DS -> next SS.
            #
            # Incoming foot becomes new support.
            # ------------------------------------------------

            support_index += 1


            phase = (
                get_next_gait_phase(
                    phase
                )
            )


        else:

            raise RuntimeError(
                f"Unsupported phase: {phase}"
            )


        first_segment = False


    return segments


# ============================================================
# EVALUATE ONE SEGMENT
# ============================================================

def _evaluate_segment(
    segment,
    local_time,
    omega0
):
    if segment[
        "kind"
    ] == "SS":

        return evaluate_ss_dcm(

            xi_start=segment[
                "xi_start"
            ],

            W_p_support_xy=segment[
                "support"
            ],

            local_time=local_time,

            omega0=omega0
        )


    if segment[
        "kind"
    ] == "DS":

        return evaluate_ds_dcm(

            xi_start=segment[
                "xi_start"
            ],

            xi_dot_start=segment[
                "xi_dot_start"
            ],

            xi_end=segment[
                "xi_end"
            ],

            xi_dot_end=segment[
                "xi_dot_end"
            ],

            local_time=local_time,

            duration=segment[
                "duration"
            ]
        )


    raise RuntimeError(
        "Unknown DCM segment type."
    )


# ============================================================
# SAMPLE PIECEWISE DCM TRAJECTORY
# ============================================================

def sample_dcm_segments(
    xi_current,
    segments,
    omega0,
    sample_times
):
    xi_current = _validate_vector(
        xi_current,
        "xi_current",
        2
    )


    sample_times = np.asarray(
        sample_times,
        dtype=float
    ).reshape(-1)


    if not np.all(
        np.isfinite(
            sample_times
        )
    ):

        raise ValueError(
            "sample_times chứa NaN hoặc Inf."
        )


    xi_samples = np.zeros(
        (
            sample_times.size,
            2
        ),
        dtype=float
    )


    xi_dot_samples = np.zeros(
        (
            sample_times.size,
            2
        ),
        dtype=float
    )


    for i, sample_time in enumerate(
        sample_times
    ):

        # ----------------------------------------------------
        # Current instant.
        # ----------------------------------------------------

        if abs(
            sample_time
        ) <= 1e-12:

            xi_samples[
                i
            ] = (
                xi_current
            )


            if len(
                segments
            ) > 0:

                (
                    _,
                    xi_dot

                ) = _evaluate_segment(

                    segment=segments[0],

                    local_time=0.0,

                    omega0=omega0
                )


                xi_dot_samples[
                    i
                ] = (
                    xi_dot
                )


            continue


        found = False


        for segment in segments:

            if (
                sample_time
                >=
                segment[
                    "t_start"
                ]
                -
                1e-12

                and

                sample_time
                <=
                segment[
                    "t_end"
                ]
                +
                1e-12
            ):

                local_time = (

                    sample_time

                    -
                    segment[
                        "t_start"
                    ]
                )


                local_time = float(
                    np.clip(
                        local_time,
                        0.0,
                        segment[
                            "duration"
                        ]
                    )
                )


                (
                    xi,
                    xi_dot

                ) = _evaluate_segment(

                    segment=segment,

                    local_time=local_time,

                    omega0=omega0
                )


                xi_samples[
                    i
                ] = (
                    xi
                )


                xi_dot_samples[
                    i
                ] = (
                    xi_dot
                )


                found = True

                break


        if not found:

            raise RuntimeError(
                "DCM segments không phủ hết sample_times."
            )


    return (
        xi_samples,
        xi_dot_samples
    )


# ============================================================
# BUILD DCM REFERENCE
# ============================================================

def build_dcm_reference(
    current_phase,
    t_phase,
    timing,
    W_p_CoM,
    W_v_CoM,
    support_points,
    com_height=None,
    gravity=DEFAULT_GRAVITY,
    horizon=None,
    dt_mpc=None
):
    """
    Build DCM reference from the CURRENT measured CoM state.

    Returns N+1 samples:

        index 0:
            current time k

        index 1:
            k + dt_mpc

        ...

        index N:
            k + N*dt_mpc
    """


    W_p_CoM = _validate_vector(
        W_p_CoM,
        "W_p_CoM",
        3
    )


    W_v_CoM = _validate_vector(
        W_v_CoM,
        "W_v_CoM",
        3
    )


    if com_height is None:

        com_height = float(
            W_p_CoM[
                2
            ]
        )


    omega0 = (
        compute_omega0(
            com_height=com_height,
            gravity=gravity
        )
    )


    xi_current = (
        compute_dcm_xy(

            W_p_CoM=W_p_CoM,

            W_v_CoM=W_v_CoM,

            omega0=omega0
        )
    )


    if horizon is None:

        horizon = int(
            timing[
                "horizon"
            ]
        )


    if dt_mpc is None:

        dt_mpc = float(
            timing[
                "dt_mpc"
            ]
        )


    segments = (
        build_dcm_segments(

            current_phase=current_phase,

            t_phase=t_phase,

            timing=timing,

            xi_current=xi_current,

            support_points=support_points,

            omega0=omega0,

            horizon=horizon,

            dt_mpc=dt_mpc
        )
    )


    sample_times = (

        np.arange(
            horizon + 1,
            dtype=float
        )

        *

        dt_mpc
    )


    (
        xi_samples,
        xi_dot_samples

    ) = sample_dcm_segments(

        xi_current=xi_current,

        segments=segments,

        omega0=omega0,

        sample_times=sample_times
    )


    return {

        "omega0":
            omega0,

        "sample_times":
            sample_times,

        "xi":
            xi_samples,

        "xi_dot":
            xi_dot_samples,

        "segments":
            segments,
    }


# ============================================================
# BUILD CoM REFERENCE FROM DCM
#
# Flat-ground relation:
#
#   p_dot_CoM
#       =
#   omega0
#   (xi - p_CoM)
#
#
# Integration:
#
# Heun predictor-corrector.
# ============================================================

def build_com_reference_from_dcm(
    W_p_CoM_current,
    xi_samples,
    omega0,
    dt_mpc,
    com_height
):
    W_p_CoM_current = _validate_vector(
        W_p_CoM_current,
        "W_p_CoM_current",
        3
    )


    xi_samples = np.asarray(
        xi_samples,
        dtype=float
    )


    if (
        xi_samples.ndim != 2
        or
        xi_samples.shape[1] != 2
    ):

        raise ValueError(
            "xi_samples phải có shape (N+1, 2)."
        )


    N = (
        xi_samples.shape[
            0
        ]
        -
        1
    )


    p_xy = np.zeros(
        (
            N + 1,
            2
        ),
        dtype=float
    )


    v_xy = np.zeros(
        (
            N + 1,
            2
        ),
        dtype=float
    )


    p_xy[
        0
    ] = (
        W_p_CoM_current[
            0:2
        ]
    )


    v_xy[
        0
    ] = (

        omega0

        *

        (
            xi_samples[
                0
            ]
            -
            p_xy[
                0
            ]
        )
    )


    for j in range(
        N
    ):

        # ----------------------------------------------------
        # Predictor
        # ----------------------------------------------------

        v_start = (

            omega0

            *

            (
                xi_samples[
                    j
                ]

                -

                p_xy[
                    j
                ]
            )
        )


        p_predict = (

            p_xy[
                j
            ]

            +

            dt_mpc
            *
            v_start
        )


        # ----------------------------------------------------
        # Corrector
        # ----------------------------------------------------

        v_end_predict = (

            omega0

            *

            (
                xi_samples[
                    j + 1
                ]

                -

                p_predict
            )
        )


        p_xy[
            j + 1
        ] = (

            p_xy[
                j
            ]

            +

            0.5
            *
            dt_mpc
            *
            (
                v_start
                +
                v_end_predict
            )
        )


        # ----------------------------------------------------
        # Consistent DCM-CoM velocity relation
        # ----------------------------------------------------

        v_xy[
            j + 1
        ] = (

            omega0

            *

            (
                xi_samples[
                    j + 1
                ]

                -

                p_xy[
                    j + 1
                ]
            )
        )


    # ========================================================
    # 3D OUTPUT
    # ========================================================

    W_p_CoM_ref = np.zeros(
        (
            N + 1,
            3
        ),
        dtype=float
    )


    W_v_CoM_ref = np.zeros(
        (
            N + 1,
            3
        ),
        dtype=float
    )


    W_p_CoM_ref[
        :,
        0:2
    ] = (
        p_xy
    )


    W_p_CoM_ref[
        :,
        2
    ] = (
        float(
            com_height
        )
    )


    W_v_CoM_ref[
        :,
        0:2
    ] = (
        v_xy
    )


    # Flat-ground constant-height command:
    #
    #   v_z_ref = 0
    W_v_CoM_ref[
        :,
        2
    ] = 0.0


    return (
        W_p_CoM_ref,
        W_v_CoM_ref
    )


# ============================================================
# BUILD COMPLETE MPC STATE REFERENCE
#
# Current SRBD state:
#
#   x =
#
#   [
#       p_CoM
#       euler
#       v_CoM
#       omega_body
#       1
#   ]
#
#   in R^13
#
#
# Prediction model:
#
#   X =
#
#   [
#       x_{k+1}
#       x_{k+2}
#       ...
#       x_{k+N}
#   ]
#
#
# Therefore X_ref contains FUTURE samples 1...N,
# not current sample 0.
# ============================================================

def build_mpc_reference(
    current_phase,
    t_phase,
    timing,
    W_p_CoM,
    W_v_CoM,
    support_points,
    com_height=None,
    desired_euler=None,
    desired_angular_velocity=None,
    gravity=DEFAULT_GRAVITY,
    horizon=None,
    dt_mpc=None
):
    """
    Build complete SRBD-MPC state reference.

    Returns
    -------
    reference : dict

        reference["X_ref"]
            shape (13*N,)

        reference["X_ref_matrix"]
            shape (N,13)

        reference["W_p_CoM_ref"]
            shape (N,3)

        reference["W_v_CoM_ref"]
            shape (N,3)

        reference["xi_ref"]
            shape (N,2)

        reference["xi_dot_ref"]
            shape (N,2)

        reference["dcm"]
            complete DCM diagnostics
    """


    W_p_CoM = _validate_vector(
        W_p_CoM,
        "W_p_CoM",
        3
    )


    W_v_CoM = _validate_vector(
        W_v_CoM,
        "W_v_CoM",
        3
    )


    if com_height is None:

        com_height = float(
            W_p_CoM[
                2
            ]
        )


    if desired_euler is None:

        desired_euler = np.zeros(
            3,
            dtype=float
        )


    if desired_angular_velocity is None:

        desired_angular_velocity = np.zeros(
            3,
            dtype=float
        )


    desired_euler = _validate_vector(
        desired_euler,
        "desired_euler",
        3
    )


    desired_angular_velocity = _validate_vector(
        desired_angular_velocity,
        "desired_angular_velocity",
        3
    )


    if horizon is None:

        horizon = int(
            timing[
                "horizon"
            ]
        )


    if dt_mpc is None:

        dt_mpc = float(
            timing[
                "dt_mpc"
            ]
        )


    # ========================================================
    # 1. DCM
    # ========================================================

    dcm = build_dcm_reference(

        current_phase=current_phase,

        t_phase=t_phase,

        timing=timing,

        W_p_CoM=W_p_CoM,

        W_v_CoM=W_v_CoM,

        support_points=support_points,

        com_height=com_height,

        gravity=gravity,

        horizon=horizon,

        dt_mpc=dt_mpc
    )


    omega0 = (
        dcm[
            "omega0"
        ]
    )


    # ========================================================
    # 2. CoM
    # ========================================================

    (
        W_p_CoM_all,
        W_v_CoM_all

    ) = build_com_reference_from_dcm(

        W_p_CoM_current=W_p_CoM,

        xi_samples=dcm[
            "xi"
        ],

        omega0=omega0,

        dt_mpc=dt_mpc,

        com_height=com_height
    )


    # --------------------------------------------------------
    # MPC predicts:
    #
    #   x_{k+1} ... x_{k+N}
    #
    # so remove index 0.
    # --------------------------------------------------------

    W_p_CoM_ref = (
        W_p_CoM_all[
            1:
        ].copy()
    )


    W_v_CoM_ref = (
        W_v_CoM_all[
            1:
        ].copy()
    )


    xi_ref = (
        dcm[
            "xi"
        ][
            1:
        ].copy()
    )


    xi_dot_ref = (
        dcm[
            "xi_dot"
        ][
            1:
        ].copy()
    )


    # ========================================================
    # 3. COMPLETE x_ref MATRIX
    # ========================================================

    X_ref_matrix = np.zeros(
        (
            horizon,
            13
        ),
        dtype=float
    )


    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    X_ref_matrix[
        :,
        0:3
    ] = (
        W_p_CoM_ref
    )


    # --------------------------------------------------------
    # Euler
    # --------------------------------------------------------

    X_ref_matrix[
        :,
        3:6
    ] = (
        desired_euler
    )


    # --------------------------------------------------------
    # Linear velocity
    # --------------------------------------------------------

    X_ref_matrix[
        :,
        6:9
    ] = (
        W_v_CoM_ref
    )


    # --------------------------------------------------------
    # Angular velocity
    # --------------------------------------------------------

    X_ref_matrix[
        :,
        9:12
    ] = (
        desired_angular_velocity
    )


    # --------------------------------------------------------
    # Homogeneous state
    # --------------------------------------------------------

    X_ref_matrix[
        :,
        12
    ] = 1.0


    # ========================================================
    # 4. STACK
    #
    #   [x1, x2, ..., xN]
    # ========================================================

    X_ref = (
        X_ref_matrix.reshape(
            -1
        )
    )


    return {

        "X_ref":
            X_ref,

        "X_ref_matrix":
            X_ref_matrix,

        "W_p_CoM_ref":
            W_p_CoM_ref,

        "W_v_CoM_ref":
            W_v_CoM_ref,

        "xi_ref":
            xi_ref,

        "xi_dot_ref":
            xi_dot_ref,

        "omega0":
            omega0,

        "dcm":
            dcm,
    }