import numpy as np


# ============================================================
# DCM PRIMITIVES
# ============================================================

try:

    from .dcm_model import (
        compute_dcm_dot_xy
    )

    from .dcm_single_support import (
        evaluate_ss_dcm,
        evaluate_ss_dcm_velocity
    )

    from .dcm_double_support import (
        evaluate_ds_dcm,
        evaluate_ds_dcm_velocity
    )

except ImportError:

    from dcm_model import (
        compute_dcm_dot_xy
    )

    from dcm_single_support import (
        evaluate_ss_dcm,
        evaluate_ss_dcm_velocity
    )

    from dcm_double_support import (
        evaluate_ds_dcm,
        evaluate_ds_dcm_velocity
    )


# ============================================================
# GAIT
# ============================================================

try:

    from Command.gait_timing import (
        get_nominal_phase_duration
    )

    from Command.contact_schedule import (
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

DEFAULT_ALPHA_DS_INI = 0.5

TOL = 1e-12


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


def _validate_positive_scalar(
    value,
    name
):

    value = float(
        value
    )


    if (
        not np.isfinite(
            value
        )
        or
        value <= 0.0
    ):

        raise ValueError(
            f"{name} phải hữu hạn và > 0."
        )


    return value


def _validate_support_points(
    support_points
):
    """
    Ordered support points in walking time.

    Example starting from LSS:

        [
            current left support,
            next right support,
            next left support,
            next right support,
            ...
        ]

    Example starting from RSS:

        [
            current right support,
            next left support,
            next right support,
            ...
        ]

    Input:
        shape (M,2) or (M,3)

    Only x-y components are used.
    """

    support_points = np.asarray(
        support_points,
        dtype=float
    )


    if support_points.ndim != 2:

        raise ValueError(
            "support_points phải là ma trận 2D."
        )


    if support_points.shape[
        1
    ] not in (
        2,
        3
    ):

        raise ValueError(
            "support_points phải có shape "
            "(M,2) hoặc (M,3)."
        )


    if support_points.shape[
        0
    ] < 2:

        raise ValueError(
            "Cần ít nhất 2 support points."
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


def _validate_timing(
    timing
):

    if not isinstance(
        timing,
        dict
    ):

        raise TypeError(
            "timing phải là dictionary."
        )


    required_keys = (
        "T_step",
        "T_SS",
        "T_DS",
        "horizon",
        "dt_mpc"
    )


    for key in required_keys:

        if key not in timing:

            raise KeyError(
                f"timing thiếu key '{key}'."
            )


    T_step = _validate_positive_scalar(
        timing[
            "T_step"
        ],
        "T_step"
    )


    T_SS = _validate_positive_scalar(
        timing[
            "T_SS"
        ],
        "T_SS"
    )


    T_DS = _validate_positive_scalar(
        timing[
            "T_DS"
        ],
        "T_DS"
    )


    if not np.isclose(
        T_step,
        T_SS + T_DS,
        atol=1e-12,
        rtol=1e-12
    ):

        raise ValueError(
            "Phải thỏa T_step = T_SS + T_DS."
        )


    return timing


def _validate_alpha_ds(
    alpha_ds_ini
):

    alpha_ds_ini = float(
        alpha_ds_ini
    )


    if (
        not np.isfinite(
            alpha_ds_ini
        )
        or
        alpha_ds_ini <= 0.0
        or
        alpha_ds_ini >= 1.0
    ):

        raise ValueError(
            "alpha_ds_ini phải nằm trong (0,1)."
        )


    return alpha_ds_ini


# ============================================================
# PRELIMINARY DCM BOUNDARIES
#
# Flat-ground specialization of paper Eq. (21).
#
# support_points:
#
#     p_0, p_1, ..., p_M
#
#
# Terminal condition:
#
#     xi_terminal = p_last
#
#
# Backward recursion:
#
#     xi_ini,i
#
#       =
#
#     p_i
#
#       +
#
#     exp(-omega0*T_step)
#
#     (xi_ini,i+1 - p_i)
#
#
# The DCM at the instantaneous transition:
#
#     p_i -> p_i+1
#
# is:
#
#     xi_transition,i = xi_ini,i+1
# ============================================================

def compute_preliminary_dcm_boundaries(
    support_points,
    T_step,
    omega0,
    W_xi_terminal_xy=None
):

    support_points = (
        _validate_support_points(
            support_points
        )
    )


    T_step = _validate_positive_scalar(
        T_step,
        "T_step"
    )


    omega0 = _validate_positive_scalar(
        omega0,
        "omega0"
    )


    M = support_points.shape[
        0
    ]


    # ========================================================
    # TERMINAL CONDITION
    # ========================================================

    if W_xi_terminal_xy is None:

        W_xi_terminal_xy = (
            support_points[
                -1
            ].copy()
        )

    else:

        W_xi_terminal_xy = (
            _validate_vector(
                W_xi_terminal_xy,
                "W_xi_terminal_xy",
                2
            )
        )


    # ========================================================
    # BACKWARD RECURSION
    # ========================================================

    W_xi_ini = np.zeros(
        (
            M,
            2
        ),
        dtype=float
    )


    W_xi_ini[
        -1
    ] = W_xi_terminal_xy


    decay = np.exp(
        -omega0
        *
        T_step
    )


    for i in range(
        M - 2,
        -1,
        -1
    ):

        W_xi_ini[
            i
        ] = (

            support_points[
                i
            ]

            +

            decay

            *

            (
                W_xi_ini[
                    i + 1
                ]

                -

                support_points[
                    i
                ]
            )
        )


    # Transition from support i to support i+1.
    W_xi_transition = (
        W_xi_ini[
            1:
        ].copy()
    )


    return {
        "support_points":
            support_points,

        "W_xi_ini":
            W_xi_ini,

        "W_xi_transition":
            W_xi_transition,

        "W_xi_terminal_xy":
            W_xi_terminal_xy.copy(),
    }


# ============================================================
# CONTINUOUS DOUBLE SUPPORT BOUNDARIES
#
# Flat-ground specialization of paper Eq. (25) and (26).
#
# For transition:
#
#     p_i -> p_i+1
#
# preliminary transition DCM:
#
#     xi_transition
#
#
# DS starts before the preliminary instantaneous transition:
#
#     Delta_t_DS_ini
#       =
#     alpha_DS_ini * T_DS
#
#
# DS ends after it:
#
#     Delta_t_DS_end
#       =
#     (1-alpha_DS_ini) * T_DS
#
# ============================================================

def compute_cds_boundaries(
    support_points,
    W_xi_transition,
    T_DS,
    omega0,
    alpha_ds_ini=DEFAULT_ALPHA_DS_INI
):

    support_points = (
        _validate_support_points(
            support_points
        )
    )


    W_xi_transition = np.asarray(
        W_xi_transition,
        dtype=float
    )


    expected_shape = (
        support_points.shape[
            0
        ] - 1,
        2
    )


    if W_xi_transition.shape != expected_shape:

        raise ValueError(
            "W_xi_transition phải có shape "
            f"{expected_shape}, "
            f"nhưng nhận được {W_xi_transition.shape}."
        )


    T_DS = _validate_positive_scalar(
        T_DS,
        "T_DS"
    )


    omega0 = _validate_positive_scalar(
        omega0,
        "omega0"
    )


    alpha_ds_ini = (
        _validate_alpha_ds(
            alpha_ds_ini
        )
    )


    delta_t_ds_ini = (
        alpha_ds_ini
        *
        T_DS
    )


    delta_t_ds_end = (
        (
            1.0
            -
            alpha_ds_ini
        )
        *
        T_DS
    )


    number_of_transitions = (
        support_points.shape[
            0
        ] - 1
    )


    W_xi_ds_start = np.zeros(
        (
            number_of_transitions,
            2
        ),
        dtype=float
    )


    W_xi_dot_ds_start = np.zeros_like(
        W_xi_ds_start
    )


    W_xi_ds_end = np.zeros_like(
        W_xi_ds_start
    )


    W_xi_dot_ds_end = np.zeros_like(
        W_xi_ds_start
    )


    # ========================================================
    # EACH SUPPORT TRANSITION
    # ========================================================

    for i in range(
        number_of_transitions
    ):

        W_p_old = (
            support_points[
                i
            ]
        )


        W_p_new = (
            support_points[
                i + 1
            ]
        )


        W_xi_transition_i = (
            W_xi_transition[
                i
            ]
        )


        # ----------------------------------------------------
        # DS START
        #
        # Move backward from instantaneous transition using
        # old support.
        # ----------------------------------------------------

        W_xi_ds_start[
            i
        ] = (

            W_p_old

            +

            np.exp(
                -omega0
                *
                delta_t_ds_ini
            )

            *

            (
                W_xi_transition_i
                -
                W_p_old
            )
        )


        # ----------------------------------------------------
        # DS END
        #
        # Move forward from instantaneous transition using
        # new support.
        # ----------------------------------------------------

        W_xi_ds_end[
            i
        ] = (

            W_p_new

            +

            np.exp(
                omega0
                *
                delta_t_ds_end
            )

            *

            (
                W_xi_transition_i
                -
                W_p_new
            )
        )


        # ----------------------------------------------------
        # Boundary velocities
        # ----------------------------------------------------

        W_xi_dot_ds_start[
            i
        ] = compute_dcm_dot_xy(

            W_xi_xy=(
                W_xi_ds_start[
                    i
                ]
            ),

            W_p_support_xy=(
                W_p_old
            ),

            omega0=omega0
        )


        W_xi_dot_ds_end[
            i
        ] = compute_dcm_dot_xy(

            W_xi_xy=(
                W_xi_ds_end[
                    i
                ]
            ),

            W_p_support_xy=(
                W_p_new
            ),

            omega0=omega0
        )


    return {
        "alpha_ds_ini":
            alpha_ds_ini,

        "delta_t_ds_ini":
            delta_t_ds_ini,

        "delta_t_ds_end":
            delta_t_ds_end,

        "W_xi_ds_start":
            W_xi_ds_start,

        "W_xi_dot_ds_start":
            W_xi_dot_ds_start,

        "W_xi_ds_end":
            W_xi_ds_end,

        "W_xi_dot_ds_end":
            W_xi_dot_ds_end,
    }


# ============================================================
# BUILD COMPLETE NOMINAL DCM PLAN
#
# This combines:
#
#     backward recursion
#
# and:
#
#     CDS boundary generation
#
#
# For every support i before the final support, compute the
# DCM at the beginning of its SS phase.
#
#
# Since SS lasts T_SS:
#
#     xi_SS_start
#
#       =
#
#     p_i
#
#       +
#
#     exp(-omega0*T_SS)
#
#     (xi_DS_start - p_i)
#
#
# This construction guarantees:
#
#     previous DS end
#
#       =
#
#     next SS start
#
# when:
#
#     T_step = T_SS + T_DS
# ============================================================

def build_nominal_dcm_plan(
    support_points,
    timing,
    omega0,
    alpha_ds_ini=DEFAULT_ALPHA_DS_INI,
    W_xi_terminal_xy=None
):

    timing = _validate_timing(
        timing
    )


    support_points = (
        _validate_support_points(
            support_points
        )
    )


    omega0 = _validate_positive_scalar(
        omega0,
        "omega0"
    )


    alpha_ds_ini = (
        _validate_alpha_ds(
            alpha_ds_ini
        )
    )


    T_step = float(
        timing[
            "T_step"
        ]
    )


    T_SS = float(
        timing[
            "T_SS"
        ]
    )


    T_DS = float(
        timing[
            "T_DS"
        ]
    )


    # ========================================================
    # 1. PRELIMINARY ZIG-ZAG DCM
    # ========================================================

    preliminary = (
        compute_preliminary_dcm_boundaries(

            support_points=(
                support_points
            ),

            T_step=T_step,

            omega0=omega0,

            W_xi_terminal_xy=(
                W_xi_terminal_xy
            )
        )
    )


    # ========================================================
    # 2. CDS TRANSITION BOUNDARIES
    # ========================================================

    cds = compute_cds_boundaries(

        support_points=(
            support_points
        ),

        W_xi_transition=(
            preliminary[
                "W_xi_transition"
            ]
        ),

        T_DS=T_DS,

        omega0=omega0,

        alpha_ds_ini=(
            alpha_ds_ini
        )
    )


    # ========================================================
    # 3. SINGLE-SUPPORT START BOUNDARIES
    # ========================================================

    number_of_ss_phases = (
        support_points.shape[
            0
        ] - 1
    )


    W_xi_ss_start = np.zeros(
        (
            number_of_ss_phases,
            2
        ),
        dtype=float
    )


    for i in range(
        number_of_ss_phases
    ):

        W_p_support = (
            support_points[
                i
            ]
        )


        W_xi_ss_start[
            i
        ] = (

            W_p_support

            +

            np.exp(
                -omega0
                *
                T_SS
            )

            *

            (
                cds[
                    "W_xi_ds_start"
                ][
                    i
                ]

                -

                W_p_support
            )
        )


    return {
        "support_points":
            support_points,

        "W_xi_ini_preliminary":
            preliminary[
                "W_xi_ini"
            ],

        "W_xi_transition":
            preliminary[
                "W_xi_transition"
            ],

        "W_xi_terminal_xy":
            preliminary[
                "W_xi_terminal_xy"
            ],

        "W_xi_ss_start":
            W_xi_ss_start,

        "W_xi_ds_start":
            cds[
                "W_xi_ds_start"
            ],

        "W_xi_dot_ds_start":
            cds[
                "W_xi_dot_ds_start"
            ],

        "W_xi_ds_end":
            cds[
                "W_xi_ds_end"
            ],

        "W_xi_dot_ds_end":
            cds[
                "W_xi_dot_ds_end"
            ],

        "alpha_ds_ini":
            cds[
                "alpha_ds_ini"
            ],

        "delta_t_ds_ini":
            cds[
                "delta_t_ds_ini"
            ],

        "delta_t_ds_end":
            cds[
                "delta_t_ds_end"
            ],

        "T_step":
            T_step,

        "T_SS":
            T_SS,

        "T_DS":
            T_DS,

        "omega0":
            omega0,
    }


# ============================================================
# BUILD PIECEWISE NOMINAL DCM SEGMENTS
#
# IMPORTANT:
#
# No measured / actual DCM is used here.
#
# Inputs are:
#
#     support plan
#     gait timing
#     synchronized nominal phase
#
# Actual-contact delay / replanning is handled by the
# higher-level integration layer.
# ============================================================

def build_dcm_segments(
    current_phase,
    t_phase,
    timing,
    support_points,
    omega0,
    horizon=None,
    dt_mpc=None,
    alpha_ds_ini=DEFAULT_ALPHA_DS_INI,
    W_xi_terminal_xy=None
):

    current_phase = normalize_gait_phase(
        current_phase
    )


    timing = _validate_timing(
        timing
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


    T_current_phase = (
        get_nominal_phase_duration(
            current_phase,
            timing
        )
    )


    # --------------------------------------------------------
    # Delayed-contact / overdue handling belongs outside this
    # nominal DCM generator.
    # --------------------------------------------------------

    if (
        t_phase
        >
        T_current_phase
        +
        TOL
    ):

        raise ValueError(
            "t_phase vượt quá nominal phase duration. "
            "Delayed-contact / overdue phase phải được xử lý "
            "ở tầng synchronization/replanning."
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


    dt_mpc = _validate_positive_scalar(
        dt_mpc,
        "dt_mpc"
    )


    horizon_time = (
        horizon
        *
        dt_mpc
    )


    # ========================================================
    # NOMINAL DCM PLAN
    # ========================================================

    plan = build_nominal_dcm_plan(

        support_points=(
            support_points
        ),

        timing=timing,

        omega0=omega0,

        alpha_ds_ini=(
            alpha_ds_ini
        ),

        W_xi_terminal_xy=(
            W_xi_terminal_xy
        )
    )


    support_points = plan[
        "support_points"
    ]


    T_SS = plan[
        "T_SS"
    ]


    T_DS = plan[
        "T_DS"
    ]


    # ========================================================
    # WALK FORWARD THROUGH NOMINAL PHASES
    # ========================================================

    phase = (
        current_phase
    )


    phase_time = min(
        t_phase,
        T_current_phase
    )


    # support_index always denotes:
    #
    #   current SS support
    #
    # or:
    #
    #   outgoing support during DS.
    support_index = 0


    total_time = 0.0


    segments = []


    while (
        total_time
        <
        horizon_time
        -
        TOL
    ):

        # ====================================================
        # SINGLE SUPPORT
        # ====================================================

        if phase in (
            "LSS",
            "RSS"
        ):

            if support_index >= (
                plan[
                    "W_xi_ss_start"
                ].shape[
                    0
                ]
            ):

                raise ValueError(
                    "Không đủ future support points "
                    "để phủ DCM horizon."
                )


            remaining_phase_time = (
                T_SS
                -
                phase_time
            )


            # Current phase already exactly at boundary.
            if remaining_phase_time <= TOL:

                phase = get_next_gait_phase(
                    phase
                )

                phase_time = 0.0

                continue


            remaining_horizon_time = (
                horizon_time
                -
                total_time
            )


            segment_duration = min(
                remaining_phase_time,
                remaining_horizon_time
            )


            W_p_support = (
                support_points[
                    support_index
                ]
            )


            W_xi_full_ss_start = (
                plan[
                    "W_xi_ss_start"
                ][
                    support_index
                ]
            )


            W_xi_start = evaluate_ss_dcm(

                W_xi_initial_xy=(
                    W_xi_full_ss_start
                ),

                W_p_support_xy=(
                    W_p_support
                ),

                time=phase_time,

                omega0=omega0
            )


            W_xi_dot_start = (
                evaluate_ss_dcm_velocity(

                    W_xi_initial_xy=(
                        W_xi_full_ss_start
                    ),

                    W_p_support_xy=(
                        W_p_support
                    ),

                    time=phase_time,

                    omega0=omega0
                )
            )


            W_xi_end = evaluate_ss_dcm(

                W_xi_initial_xy=(
                    W_xi_full_ss_start
                ),

                W_p_support_xy=(
                    W_p_support
                ),

                time=(
                    phase_time
                    +
                    segment_duration
                ),

                omega0=omega0
            )


            W_xi_dot_end = (
                evaluate_ss_dcm_velocity(

                    W_xi_initial_xy=(
                        W_xi_full_ss_start
                    ),

                    W_p_support_xy=(
                        W_p_support
                    ),

                    time=(
                        phase_time
                        +
                        segment_duration
                    ),

                    omega0=omega0
                )
            )


            segments.append(
                {
                    "kind":
                        "SS",

                    "phase":
                        phase,

                    "support_index":
                        support_index,

                    "t_start":
                        total_time,

                    "t_end":
                        total_time
                        +
                        segment_duration,

                    "duration":
                        segment_duration,

                    "phase_time_start":
                        phase_time,

                    "phase_time_end":
                        phase_time
                        +
                        segment_duration,

                    "W_p_support_xy":
                        W_p_support.copy(),

                    "W_xi_start_xy":
                        W_xi_start.copy(),

                    "W_xi_dot_start_xy":
                        W_xi_dot_start.copy(),

                    "W_xi_end_xy":
                        W_xi_end.copy(),

                    "W_xi_dot_end_xy":
                        W_xi_dot_end.copy(),
                }
            )


            total_time += (
                segment_duration
            )


            phase_time += (
                segment_duration
            )


            if (
                phase_time
                >=
                T_SS
                -
                TOL
            ):

                phase = get_next_gait_phase(
                    phase
                )

                phase_time = 0.0


        # ====================================================
        # DOUBLE SUPPORT
        # ====================================================

        elif phase in (
            "DS_L_TO_R",
            "DS_R_TO_L"
        ):

            if support_index >= (
                plan[
                    "W_xi_ds_start"
                ].shape[
                    0
                ]
            ):

                raise ValueError(
                    "Không đủ future support points "
                    "cho DS transition."
                )


            remaining_phase_time = (
                T_DS
                -
                phase_time
            )


            if remaining_phase_time <= TOL:

                support_index += 1

                phase = get_next_gait_phase(
                    phase
                )

                phase_time = 0.0

                continue


            remaining_horizon_time = (
                horizon_time
                -
                total_time
            )


            segment_duration = min(
                remaining_phase_time,
                remaining_horizon_time
            )


            phase_time_end = (
                phase_time
                +
                segment_duration
            )


            # ------------------------------------------------
            # Evaluate ORIGINAL full CDS polynomial at both
            # ends of this possibly partial DS interval.
            # ------------------------------------------------

            W_xi_start = evaluate_ds_dcm(

                W_xi_start_xy=(
                    plan[
                        "W_xi_ds_start"
                    ][
                        support_index
                    ]
                ),

                W_xi_dot_start_xy=(
                    plan[
                        "W_xi_dot_ds_start"
                    ][
                        support_index
                    ]
                ),

                W_xi_end_xy=(
                    plan[
                        "W_xi_ds_end"
                    ][
                        support_index
                    ]
                ),

                W_xi_dot_end_xy=(
                    plan[
                        "W_xi_dot_ds_end"
                    ][
                        support_index
                    ]
                ),

                time=phase_time,

                T_DS=T_DS
            )


            W_xi_dot_start = (
                evaluate_ds_dcm_velocity(

                    W_xi_start_xy=(
                        plan[
                            "W_xi_ds_start"
                        ][
                            support_index
                        ]
                    ),

                    W_xi_dot_start_xy=(
                        plan[
                            "W_xi_dot_ds_start"
                        ][
                            support_index
                        ]
                    ),

                    W_xi_end_xy=(
                        plan[
                            "W_xi_ds_end"
                        ][
                            support_index
                        ]
                    ),

                    W_xi_dot_end_xy=(
                        plan[
                            "W_xi_dot_ds_end"
                        ][
                            support_index
                        ]
                    ),

                    time=phase_time,

                    T_DS=T_DS
                )
            )


            W_xi_end = evaluate_ds_dcm(

                W_xi_start_xy=(
                    plan[
                        "W_xi_ds_start"
                    ][
                        support_index
                    ]
                ),

                W_xi_dot_start_xy=(
                    plan[
                        "W_xi_dot_ds_start"
                    ][
                        support_index
                    ]
                ),

                W_xi_end_xy=(
                    plan[
                        "W_xi_ds_end"
                    ][
                        support_index
                    ]
                ),

                W_xi_dot_end_xy=(
                    plan[
                        "W_xi_dot_ds_end"
                    ][
                        support_index
                    ]
                ),

                time=phase_time_end,

                T_DS=T_DS
            )


            W_xi_dot_end = (
                evaluate_ds_dcm_velocity(

                    W_xi_start_xy=(
                        plan[
                            "W_xi_ds_start"
                        ][
                            support_index
                        ]
                    ),

                    W_xi_dot_start_xy=(
                        plan[
                            "W_xi_dot_ds_start"
                        ][
                            support_index
                        ]
                    ),

                    W_xi_end_xy=(
                        plan[
                            "W_xi_ds_end"
                        ][
                            support_index
                        ]
                    ),

                    W_xi_dot_end_xy=(
                        plan[
                            "W_xi_dot_ds_end"
                        ][
                            support_index
                        ]
                    ),

                    time=phase_time_end,

                    T_DS=T_DS
                )
            )


            segments.append(
                {
                    "kind":
                        "DS",

                    "phase":
                        phase,

                    "support_index":
                        support_index,

                    "t_start":
                        total_time,

                    "t_end":
                        total_time
                        +
                        segment_duration,

                    "duration":
                        segment_duration,

                    "phase_time_start":
                        phase_time,

                    "phase_time_end":
                        phase_time_end,

                    "W_p_old_support_xy":
                        support_points[
                            support_index
                        ].copy(),

                    "W_p_new_support_xy":
                        support_points[
                            support_index
                            +
                            1
                        ].copy(),

                    "W_xi_start_xy":
                        W_xi_start.copy(),

                    "W_xi_dot_start_xy":
                        W_xi_dot_start.copy(),

                    "W_xi_end_xy":
                        W_xi_end.copy(),

                    "W_xi_dot_end_xy":
                        W_xi_dot_end.copy(),
                }
            )


            total_time += (
                segment_duration
            )


            phase_time += (
                segment_duration
            )


            if (
                phase_time
                >=
                T_DS
                -
                TOL
            ):

                support_index += 1

                phase = get_next_gait_phase(
                    phase
                )

                phase_time = 0.0


        else:

            raise RuntimeError(
                f"Unsupported gait phase: {phase}"
            )


    return (
        segments,
        plan
    )


# ============================================================
# EVALUATE ONE GENERATED SEGMENT
# ============================================================

def evaluate_dcm_segment(
    segment,
    local_time,
    omega0
):

    local_time = float(
        local_time
    )


    duration = float(
        segment[
            "duration"
        ]
    )


    if (
        local_time < -TOL
        or
        local_time > duration + TOL
    ):

        raise ValueError(
            "local_time nằm ngoài DCM segment."
        )


    local_time = float(
        np.clip(
            local_time,
            0.0,
            duration
        )
    )


    # ========================================================
    # SS
    # ========================================================

    if segment[
        "kind"
    ] == "SS":

        W_xi = evaluate_ss_dcm(

            W_xi_initial_xy=(
                segment[
                    "W_xi_start_xy"
                ]
            ),

            W_p_support_xy=(
                segment[
                    "W_p_support_xy"
                ]
            ),

            time=local_time,

            omega0=omega0
        )


        W_xi_dot = (
            evaluate_ss_dcm_velocity(

                W_xi_initial_xy=(
                    segment[
                        "W_xi_start_xy"
                    ]
                ),

                W_p_support_xy=(
                    segment[
                        "W_p_support_xy"
                    ]
                ),

                time=local_time,

                omega0=omega0
            )
        )


        return (
            W_xi,
            W_xi_dot
        )


    # ========================================================
    # DS
    #
    # A cropped interval of a cubic remains a cubic.
    # Its position/velocity at both ends uniquely determine it.
    # ========================================================

    if segment[
        "kind"
    ] == "DS":

        W_xi = evaluate_ds_dcm(

            W_xi_start_xy=(
                segment[
                    "W_xi_start_xy"
                ]
            ),

            W_xi_dot_start_xy=(
                segment[
                    "W_xi_dot_start_xy"
                ]
            ),

            W_xi_end_xy=(
                segment[
                    "W_xi_end_xy"
                ]
            ),

            W_xi_dot_end_xy=(
                segment[
                    "W_xi_dot_end_xy"
                ]
            ),

            time=local_time,

            T_DS=duration
        )


        W_xi_dot = (
            evaluate_ds_dcm_velocity(

                W_xi_start_xy=(
                    segment[
                        "W_xi_start_xy"
                    ]
                ),

                W_xi_dot_start_xy=(
                    segment[
                        "W_xi_dot_start_xy"
                    ]
                ),

                W_xi_end_xy=(
                    segment[
                        "W_xi_end_xy"
                    ]
                ),

                W_xi_dot_end_xy=(
                    segment[
                        "W_xi_dot_end_xy"
                    ]
                ),

                time=local_time,

                T_DS=duration
            )
        )


        return (
            W_xi,
            W_xi_dot
        )


    raise RuntimeError(
        "Unknown DCM segment kind."
    )


# ============================================================
# SAMPLE SEGMENTS
# ============================================================

def sample_dcm_segments(
    segments,
    omega0,
    sample_times
):

    if not isinstance(
        segments,
        list
    ):

        raise TypeError(
            "segments phải là list."
        )


    if len(
        segments
    ) == 0:

        raise ValueError(
            "segments không được rỗng."
        )


    sample_times = np.asarray(
        sample_times,
        dtype=float
    ).reshape(-1)


    if (
        sample_times.size == 0
        or
        not np.all(
            np.isfinite(
                sample_times
            )
        )
    ):

        raise ValueError(
            "sample_times không hợp lệ."
        )


    if np.any(
        sample_times < 0.0
    ):

        raise ValueError(
            "sample_times không được âm."
        )


    W_xi_ref = np.zeros(
        (
            sample_times.size,
            2
        ),
        dtype=float
    )


    W_xi_dot_ref = np.zeros_like(
        W_xi_ref
    )


    phase_sequence = []

    segment_kind_sequence = []


    for j, sample_time in enumerate(
        sample_times
    ):

        selected_segment = None


        for segment in segments:

            if (
                sample_time
                >=
                segment[
                    "t_start"
                ]
                -
                TOL
                and
                sample_time
                <=
                segment[
                    "t_end"
                ]
                +
                TOL
            ):

                selected_segment = (
                    segment
                )

                break


        if selected_segment is None:

            raise RuntimeError(
                "DCM segments không phủ sample time."
            )


        local_time = (

            sample_time

            -

            selected_segment[
                "t_start"
            ]
        )


        (
            W_xi_ref[
                j
            ],

            W_xi_dot_ref[
                j
            ]

        ) = evaluate_dcm_segment(

            segment=selected_segment,

            local_time=local_time,

            omega0=omega0
        )


        phase_sequence.append(
            selected_segment[
                "phase"
            ]
        )


        segment_kind_sequence.append(
            selected_segment[
                "kind"
            ]
        )


    return {
        "W_xi_ref":
            W_xi_ref,

        "W_xi_dot_ref":
            W_xi_dot_ref,

        "phase_sequence":
            tuple(
                phase_sequence
            ),

        "segment_kind_sequence":
            tuple(
                segment_kind_sequence
            ),
    }


# ============================================================
# BUILD DCM MPC REFERENCE HORIZON
#
# Output:
#
#     xi_ref(k+1)
#     ...
#     xi_ref(k+N)
#
#
# IMPORTANT:
#
#     actual xi_k is NOT used to construct the nominal
#     trajectory.
# ============================================================

def build_dcm_reference_horizon(
    current_phase,
    t_phase,
    timing,
    support_points,
    omega0,
    horizon=None,
    dt_mpc=None,
    alpha_ds_ini=DEFAULT_ALPHA_DS_INI,
    W_xi_terminal_xy=None
):

    timing = _validate_timing(
        timing
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


    dt_mpc = _validate_positive_scalar(
        dt_mpc,
        "dt_mpc"
    )


    # ========================================================
    # 1. BUILD CONTINUOUS NOMINAL TRAJECTORY
    # ========================================================

    (
        segments,
        plan

    ) = build_dcm_segments(

        current_phase=current_phase,

        t_phase=t_phase,

        timing=timing,

        support_points=support_points,

        omega0=omega0,

        horizon=horizon,

        dt_mpc=dt_mpc,

        alpha_ds_ini=(
            alpha_ds_ini
        ),

        W_xi_terminal_xy=(
            W_xi_terminal_xy
        )
    )


    # ========================================================
    # 2. NOMINAL DCM AT CURRENT TIME
    # ========================================================

    (
        W_xi_nominal_current_xy,
        W_xi_dot_nominal_current_xy

    ) = evaluate_dcm_segment(

        segment=segments[
            0
        ],

        local_time=0.0,

        omega0=omega0
    )


    # ========================================================
    # 3. MPC FUTURE SAMPLE TIMES
    #
    # Current sample k is not included.
    # ========================================================

    sample_times = (

        np.arange(
            1,
            horizon + 1,
            dtype=float
        )

        *

        dt_mpc
    )


    # ========================================================
    # 4. SAMPLE
    # ========================================================

    sampled = sample_dcm_segments(

        segments=segments,

        omega0=omega0,

        sample_times=sample_times
    )


    return {
        "W_xi_nominal_current_xy":
            W_xi_nominal_current_xy,

        "W_xi_dot_nominal_current_xy":
            W_xi_dot_nominal_current_xy,

        "sample_times":
            sample_times,

        "W_xi_ref":
            sampled[
                "W_xi_ref"
            ],

        "W_xi_dot_ref":
            sampled[
                "W_xi_dot_ref"
            ],

        "phase_sequence":
            sampled[
                "phase_sequence"
            ],

        "segment_kind_sequence":
            sampled[
                "segment_kind_sequence"
            ],

        "segments":
            segments,

        "plan":
            plan,

        "horizon":
            horizon,

        "dt_mpc":
            dt_mpc,
    }