import numpy as np


# ============================================================
# CONTACT / GAIT
# ============================================================

try:

    from .contact_schedule import (
        normalize_gait_phase,
        get_next_gait_phase
    )

except ImportError:

    from contact_schedule import (
        normalize_gait_phase,
        get_next_gait_phase
    )


# ============================================================
# VALIDATION
# ============================================================

def _validate_position3(
    value,
    name
):
    value = np.asarray(
        value,
        dtype=float
    ).reshape(-1)


    if value.shape != (
        3,
    ):

        raise ValueError(
            f"{name} phải có shape (3,), "
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
    Ordered support-point sequence.

    Convention
    ----------

    support_points[0]
        =
    current / outgoing support.


    support_points[1]
        =
    next support.


    support_points[2]
        =
    support after that.


    ...

    Each point must be expressed in World frame:

        ^W p_foot

    Shape:

        (M, 3)
    """


    support_points = np.asarray(
        support_points,
        dtype=float
    )


    if support_points.ndim != 2:

        raise ValueError(
            "support_points phải là ma trận 2D."
        )


    if support_points.shape[1] != 3:

        raise ValueError(
            "support_points phải có shape (M, 3)."
        )


    if support_points.shape[0] < 2:

        raise ValueError(
            "support_points cần ít nhất 2 điểm."
        )


    if not np.all(
        np.isfinite(
            support_points
        )
    ):

        raise ValueError(
            "support_points chứa NaN hoặc Inf."
        )


    return support_points.copy()


def _validate_phase_sequence(
    phase_sequence
):
    if not isinstance(
        phase_sequence,
        (
            list,
            tuple
        )
    ):

        raise TypeError(
            "phase_sequence phải là list hoặc tuple."
        )


    if len(
        phase_sequence
    ) == 0:

        raise ValueError(
            "phase_sequence không được rỗng."
        )


    normalized = []


    for phase in phase_sequence:

        normalized.append(
            normalize_gait_phase(
                phase
            )
        )


    return normalized


# ============================================================
# SUPPORT SIDE
#
# support_points alternate:
#
#   L, R, L, R, ...
#
# or:
#
#   R, L, R, L, ...
#
# depending on the current synchronized gait phase.
# ============================================================

def get_initial_support_foot(
    current_phase
):
    """
    Determine the side of support_points[0].

    LSS:
        support_points[0] = left support

    DS_L_TO_R:
        support_points[0] = outgoing left support

    RSS:
        support_points[0] = right support

    DS_R_TO_L:
        support_points[0] = outgoing right support
    """


    current_phase = normalize_gait_phase(
        current_phase
    )


    if current_phase in (
        "LSS",
        "DS_L_TO_R"
    ):

        return "L"


    return "R"


def get_support_foot_from_index(
    initial_support_foot,
    support_index
):
    """
    Map ordered support index to foot side.

    Example
    -------

    Initial support = L:

        index 0 -> L
        index 1 -> R
        index 2 -> L
        index 3 -> R
    """


    initial_support_foot = str(
        initial_support_foot
    ).upper()


    if initial_support_foot not in (
        "L",
        "R"
    ):

        raise ValueError(
            "initial_support_foot phải là 'L' hoặc 'R'."
        )


    support_index = int(
        support_index
    )


    if support_index < 0:

        raise ValueError(
            "support_index không được âm."
        )


    even = (
        support_index
        %
        2
        ==
        0
    )


    if initial_support_foot == "L":

        return (
            "L"
            if even
            else "R"
        )


    return (
        "R"
        if even
        else "L"
    )


# ============================================================
# SUPPORT INDEX UPDATE
#
# Incoming foot becomes the new support after:
#
#   DS_L_TO_R -> RSS
#
# or:
#
#   DS_R_TO_L -> LSS
# ============================================================

def _is_support_advance_transition(
    phase_from,
    phase_to
):
    return bool(

        (
            phase_from
            ==
            "DS_L_TO_R"

            and

            phase_to
            ==
            "RSS"
        )

        or

        (
            phase_from
            ==
            "DS_R_TO_L"

            and

            phase_to
            ==
            "LSS"
        )
    )


def advance_support_index_between_phases(
    previous_phase,
    current_phase,
    support_index
):
    """
    Advance support_index from previous_phase to current_phase.

    Normally consecutive horizon samples either:

        remain in the same phase

    or:

        move to the next nominal phase.

    This implementation also supports skipped phases when
    dt_mpc is larger than a phase duration.
    """


    previous_phase = normalize_gait_phase(
        previous_phase
    )


    current_phase = normalize_gait_phase(
        current_phase
    )


    support_index = int(
        support_index
    )


    if previous_phase == current_phase:

        return support_index


    phase = (
        previous_phase
    )


    # At most one complete four-phase gait cycle should be
    # required between neighboring MPC samples for the current
    # implementation.
    for _ in range(
        4
    ):

        next_phase = (
            get_next_gait_phase(
                phase
            )
        )


        if _is_support_advance_transition(
            phase,
            next_phase
        ):

            support_index += 1


        phase = (
            next_phase
        )


        if phase == current_phase:

            return support_index


    raise RuntimeError(
        "Không thể map phase transition trong nominal gait cycle."
    )


# ============================================================
# GET LEFT / RIGHT PLANNED POSITION FOR ONE PHASE
# ============================================================

def get_planned_feet_for_phase(
    phase,
    support_index,
    support_points,
    initial_support_foot
):
    """
    Return planned L6/R6 positions for one horizon sample.

    Important
    ---------

    For a swing foot, the next touchdown target is used.

    This is intentional for SRBD-LTV prediction because the
    swing-foot wrench is constrained to zero before contact.

    At confirmed touchdown, the corresponding support point
    should be replaced by the realized/latching foot position.
    """


    phase = normalize_gait_phase(
        phase
    )


    support_points = _validate_support_points(
        support_points
    )


    support_index = int(
        support_index
    )


    next_index = (
        support_index
        +
        1
    )


    if next_index >= (
        support_points.shape[
            0
        ]
    ):

        raise ValueError(
            "Không đủ support_points để xác định "
            "next touchdown foothold."
        )


    current_support_foot = (
        get_support_foot_from_index(

            initial_support_foot=(
                initial_support_foot
            ),

            support_index=(
                support_index
            )
        )
    )


    next_support_foot = (
        get_support_foot_from_index(

            initial_support_foot=(
                initial_support_foot
            ),

            support_index=(
                next_index
            )
        )
    )


    if (
        current_support_foot
        ==
        next_support_foot
    ):

        raise RuntimeError(
            "Ordered support sequence phải luân phiên L/R."
        )


    current_position = (
        support_points[
            support_index
        ]
    )


    next_position = (
        support_points[
            next_index
        ]
    )


    # ========================================================
    # ASSIGN CURRENT SUPPORT + NEXT TOUCHDOWN TARGET
    # ========================================================

    if current_support_foot == "L":

        W_p_L6_plan = (
            current_position.copy()
        )

        W_p_R6_plan = (
            next_position.copy()
        )


    else:

        W_p_R6_plan = (
            current_position.copy()
        )

        W_p_L6_plan = (
            next_position.copy()
        )


    # ========================================================
    # PHYSICAL CONSISTENCY CHECK
    #
    # The current phase must agree with support_index.
    # ========================================================

    if phase == "LSS":

        if current_support_foot != "L":

            raise RuntimeError(
                "LSS nhưng current support point không phải chân L."
            )


    elif phase == "RSS":

        if current_support_foot != "R":

            raise RuntimeError(
                "RSS nhưng current support point không phải chân R."
            )


    elif phase == "DS_L_TO_R":

        if current_support_foot != "L":

            raise RuntimeError(
                "DS_L_TO_R yêu cầu outgoing support là chân L."
            )


    elif phase == "DS_R_TO_L":

        if current_support_foot != "R":

            raise RuntimeError(
                "DS_R_TO_L yêu cầu outgoing support là chân R."
            )


    return (
        W_p_L6_plan,
        W_p_R6_plan
    )


# ============================================================
# BUILD COMPLETE FOOT PLAN HORIZON
# ============================================================

def build_foot_plan_horizon(
    current_phase,
    phase_sequence,
    support_points
):
    """
    Expand ordered support points into planned left/right foot
    positions over the MPC prediction horizon.

    Parameters
    ----------
    current_phase : str

        Current synchronized gait phase.


    phase_sequence : list[str]

        Horizon phase sequence.

        Normally obtained from:

            contact_schedule.build_phase_preview(...)

        Example:

            [
                "LSS",
                "LSS",
                "DS_L_TO_R",
                "DS_L_TO_R",
                "RSS",
                ...
            ]


    support_points : array_like, shape (M,3)

        Ordered sequence of support locations expressed in
        World frame.

        Example for current LSS:

            [
                current realized L support,
                next R foothold,
                next L foothold,
                next R foothold,
                ...
            ]


        Example for current DS_L_TO_R:

            [
                outgoing realized L support,
                incoming realized/planned R support,
                next L foothold,
                next R foothold,
                ...
            ]


    Returns
    -------
    result : dict

        result["W_p_L6_plan"]
            shape (N,3)

        result["W_p_R6_plan"]
            shape (N,3)

        result["support_index_sequence"]
            shape (N,)

        result["support_foot_sequence"]
            list[str]
    """


    current_phase = normalize_gait_phase(
        current_phase
    )


    phase_sequence = (
        _validate_phase_sequence(
            phase_sequence
        )
    )


    support_points = (
        _validate_support_points(
            support_points
        )
    )


    # ========================================================
    # FIRST HORIZON PHASE MUST REPRESENT CURRENT PHASE
    # ========================================================

    if phase_sequence[
        0
    ] != current_phase:

        raise ValueError(
            "phase_sequence[0] phải bằng current_phase."
        )


    N = len(
        phase_sequence
    )


    W_p_L6_plan = np.zeros(
        (
            N,
            3
        ),
        dtype=float
    )


    W_p_R6_plan = np.zeros(
        (
            N,
            3
        ),
        dtype=float
    )


    support_index_sequence = np.zeros(
        N,
        dtype=int
    )


    support_foot_sequence = []


    # ========================================================
    # INITIAL SUPPORT SIDE
    # ========================================================

    initial_support_foot = (
        get_initial_support_foot(
            current_phase
        )
    )


    support_index = 0


    previous_phase = (
        current_phase
    )


    # ========================================================
    # HORIZON LOOP
    # ========================================================

    for j, phase in enumerate(
        phase_sequence
    ):

        # ----------------------------------------------------
        # Advance support index if horizon crossed:
        #
        #   DS -> next SS
        # ----------------------------------------------------

        if j > 0:

            support_index = (
                advance_support_index_between_phases(

                    previous_phase=(
                        previous_phase
                    ),

                    current_phase=(
                        phase
                    ),

                    support_index=(
                        support_index
                    )
                )
            )


        # ----------------------------------------------------
        # Need current support point AND next touchdown target.
        # ----------------------------------------------------

        if (
            support_index
            +
            1
            >=
            support_points.shape[
                0
            ]
        ):

            raise ValueError(
                "Không đủ support_points để phủ toàn "
                "prediction horizon."
            )


        (
            W_p_L,
            W_p_R

        ) = get_planned_feet_for_phase(

            phase=phase,

            support_index=(
                support_index
            ),

            support_points=(
                support_points
            ),

            initial_support_foot=(
                initial_support_foot
            )
        )


        W_p_L6_plan[
            j
        ] = (
            W_p_L
        )


        W_p_R6_plan[
            j
        ] = (
            W_p_R
        )


        support_index_sequence[
            j
        ] = (
            support_index
        )


        support_foot_sequence.append(

            get_support_foot_from_index(

                initial_support_foot=(
                    initial_support_foot
                ),

                support_index=(
                    support_index
                )
            )
        )


        previous_phase = (
            phase
        )


    # ========================================================
    # OUTPUT
    # ========================================================

    return {

        "W_p_L6_plan":
            W_p_L6_plan,

        "W_p_R6_plan":
            W_p_R6_plan,

        "support_index_sequence":
            support_index_sequence,

        "support_foot_sequence":
            support_foot_sequence,

        "initial_support_foot":
            initial_support_foot,
    }


# ============================================================
# COMPUTE LTV MOMENT-ARM SEQUENCES
#
#   r_L,j
#       =
#   p_L6,j^plan
#       -
#   p_CoM,j^ref
#
#
#   r_R,j
#       =
#   p_R6,j^plan
#       -
#   p_CoM,j^ref
# ============================================================

def build_moment_arm_horizon(
    W_p_CoM_ref,
    W_p_L6_plan,
    W_p_R6_plan
):
    """
    Build moment-arm sequences required by LTV SRBD.
    """


    W_p_CoM_ref = np.asarray(
        W_p_CoM_ref,
        dtype=float
    )


    W_p_L6_plan = np.asarray(
        W_p_L6_plan,
        dtype=float
    )


    W_p_R6_plan = np.asarray(
        W_p_R6_plan,
        dtype=float
    )


    if (
        W_p_CoM_ref.ndim != 2
        or
        W_p_CoM_ref.shape[1] != 3
    ):

        raise ValueError(
            "W_p_CoM_ref phải có shape (N,3)."
        )


    if (
        W_p_L6_plan.shape
        !=
        W_p_CoM_ref.shape
    ):

        raise ValueError(
            "W_p_L6_plan phải có cùng shape "
            "với W_p_CoM_ref."
        )


    if (
        W_p_R6_plan.shape
        !=
        W_p_CoM_ref.shape
    ):

        raise ValueError(
            "W_p_R6_plan phải có cùng shape "
            "với W_p_CoM_ref."
        )


    if not (
        np.all(
            np.isfinite(
                W_p_CoM_ref
            )
        )
        and
        np.all(
            np.isfinite(
                W_p_L6_plan
            )
        )
        and
        np.all(
            np.isfinite(
                W_p_R6_plan
            )
        )
    ):

        raise ValueError(
            "Moment-arm inputs chứa NaN hoặc Inf."
        )


    W_r_L = (

        W_p_L6_plan

        -

        W_p_CoM_ref
    )


    W_r_R = (

        W_p_R6_plan

        -

        W_p_CoM_ref
    )


    return (
        W_r_L,
        W_r_R
    )