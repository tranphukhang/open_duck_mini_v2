from dataclasses import dataclass
from typing import Optional

import numpy as np


# ============================================================
# COMMAND MODULES
# ============================================================

from Command.contact_switch import (
    detect_contact_events,
    normalize_contact_state,
    get_contact_mode,
)

from Command.contact_schedule import (
    normalize_gait_phase,
    advance_phase_timer,
    synchronize_gait_phase,
    get_expected_transition_event,
    get_phase_contact_state,
    build_phase_preview,
)

from Command.gait_timing import (
    get_nominal_phase_duration,
)


# ============================================================
# GAIT CONVENTION
#
# LSS:
#
#     Left  = stance
#     Right = swing
#
#
# RSS:
#
#     Right = stance
#     Left  = swing
#
#
# DS_L_TO_R:
#
#     directional double support:
#
#     LSS -> RSS
#
#
# DS_R_TO_L:
#
#     directional double support:
#
#     RSS -> LSS
#
#
# IMPORTANT:
#
# [1, 1] contact alone does NOT determine DS direction.
#
# Direction is retained explicitly by current_phase.
# ============================================================


# ============================================================
# STATE
# ============================================================

@dataclass
class GaitManagerState:
    """
    Persistent gait state.

    This object contains only information that must persist
    from one simulation/control sample to the next.
    """

    current_phase: str

    t_phase: float

    previous_c_L: int

    previous_c_R: int

    # Last physical liftoff position of each foot.
    #
    # These values are latched only when:
    #
    #     LO_L
    #
    # or:
    #
    #     LO_R
    #
    # occurs.
    #
    # They must NOT be continuously overwritten while
    # the foot is swinging.
    W_p_liftoff_L: Optional[np.ndarray]

    W_p_liftoff_R: Optional[np.ndarray]


# ============================================================
# OUTPUT
# ============================================================

@dataclass
class GaitManagerOutput:
    """
    Information required by the higher-level walking loop.
    """

    # --------------------------------------------------------
    # Actual synchronized gait state
    # --------------------------------------------------------

    current_phase: str

    t_phase: float


    # --------------------------------------------------------
    # Contact
    # --------------------------------------------------------

    c_L: int

    c_R: int

    contact_mode: str

    events: dict


    # --------------------------------------------------------
    # FSM transition
    # --------------------------------------------------------

    transitioned: bool

    transition_event: Optional[str]

    expected_event: str


    # --------------------------------------------------------
    # Swing information
    # --------------------------------------------------------

    swing_foot: Optional[str]

    swing_phase: Optional[float]

    W_p_liftoff: Optional[np.ndarray]

    W_p_foothold: Optional[np.ndarray]


    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    phase_overdue: bool

    phase_contact_consistent: bool


    # --------------------------------------------------------
    # MPC future phase preview
    # --------------------------------------------------------

    phase_preview: dict


# ============================================================
# VECTOR VALIDATION
# ============================================================

def _validate_optional_vector3(
    value,
    name,
):
    """
    Validate optional 3D World-frame position.
    """

    if value is None:

        return None


    value = np.asarray(
        value,
        dtype=float,
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


    return value.copy()


# ============================================================
# GET SWING FOOT
# ============================================================

def get_swing_foot_from_phase(
    phase,
):
    """
    Return swing foot associated with synchronized gait phase.

    Mapping
    -------

        LSS
            -> R

        RSS
            -> L

        DS_L_TO_R
            -> None

        DS_R_TO_L
            -> None
    """

    phase = normalize_gait_phase(
        phase
    )


    if phase == "LSS":

        return "R"


    if phase == "RSS":

        return "L"


    return None


# ============================================================
# COMPUTE SWING PHASE
# ============================================================

def compute_swing_phase(
    current_phase,
    t_phase,
    timing,
):
    """
    Compute normalized swing phase.

        phi_swing = t_phase / T_SS

    only during single support.

    The result is clipped to:

        0 <= phi_swing <= 1

    Therefore, if touchdown is delayed beyond nominal T_SS,
    the swing reference remains frozen at the desired foothold
    while the FSM continues waiting for the actual TD event.

    Returns
    -------
    swing_phase : float or None
    """

    current_phase = normalize_gait_phase(
        current_phase
    )


    if current_phase not in (
        "LSS",
        "RSS",
    ):

        return None


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


    T_SS = float(
        timing[
            "T_SS"
        ]
    )


    if (
        not np.isfinite(
            T_SS
        )
        or
        T_SS <= 0.0
    ):

        raise ValueError(
            "timing['T_SS'] phải hữu hạn và > 0."
        )


    swing_phase = (

        t_phase

        /

        T_SS
    )


    swing_phase = float(

        np.clip(
            swing_phase,
            0.0,
            1.0,
        )
    )


    return swing_phase


# ============================================================
# SELECT CURRENT SWING LIFTOFF POSITION
# ============================================================

def _get_current_swing_liftoff(
    state,
    swing_foot,
):

    if swing_foot == "L":

        if state.W_p_liftoff_L is None:

            return None


        return state.W_p_liftoff_L.copy()


    if swing_foot == "R":

        if state.W_p_liftoff_R is None:

            return None


        return state.W_p_liftoff_R.copy()


    return None


# ============================================================
# SELECT CURRENT SWING FOOTHOLD
# ============================================================

def _get_current_swing_foothold(
    swing_foot,
    W_p_foothold_L,
    W_p_foothold_R,
):
    """
    Select externally generated foothold target.

    This function does NOT generate the foothold.
    """

    if swing_foot == "L":

        return _validate_optional_vector3(
            W_p_foothold_L,
            "W_p_foothold_L"
        )


    if swing_foot == "R":

        return _validate_optional_vector3(
            W_p_foothold_R,
            "W_p_foothold_R"
        )


    return None


# ============================================================
# INITIALIZE GAIT MANAGER
# ============================================================

def initialize_gait_manager(
    initial_phase,
    c_L,
    c_R,
    W_p_L6=None,
    W_p_R6=None,
    t_phase=0.0,
    strict_contact=True,
):
    """
    Initialize gait-manager persistent state.

    Important
    ---------

    Directional DS must be explicitly specified:

        DS_L_TO_R

    or:

        DS_R_TO_L

    It is NEVER inferred from:

        [c_L, c_R] = [1, 1]


    Parameters
    ----------
    initial_phase : str

        One of:

            LSS
            DS_L_TO_R
            RSS
            DS_R_TO_L


    c_L, c_R : {0, 1}

        Current physical contacts.


    W_p_L6, W_p_R6 : array_like or None

        Current physical foot positions.

        If initialized directly inside SS, the corresponding
        swing-foot position is used as an initial liftoff seed.

        In normal production walking, startup should normally
        enter SS through an actual LO event instead.


    t_phase : float

        Initial elapsed phase time.


    strict_contact : bool

        If True, physical contact must match initial_phase.
    """

    initial_phase = normalize_gait_phase(
        initial_phase
    )


    c_L = normalize_contact_state(
        c_L
    )


    c_R = normalize_contact_state(
        c_R
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


    W_p_L6 = _validate_optional_vector3(
        W_p_L6,
        "W_p_L6"
    )


    W_p_R6 = _validate_optional_vector3(
        W_p_R6,
        "W_p_R6"
    )


    # ========================================================
    # CONTACT CONSISTENCY
    # ========================================================

    expected_contact = (
        get_phase_contact_state(
            initial_phase
        )
    )


    actual_contact = np.array(
        [
            c_L,
            c_R,
        ],
        dtype=int,
    )


    contact_consistent = bool(

        np.array_equal(
            expected_contact,
            actual_contact,
        )
    )


    if (
        strict_contact
        and
        not contact_consistent
    ):

        raise ValueError(
            "Initial physical contact không khớp "
            f"initial_phase='{initial_phase}'. "
            f"Expected={expected_contact.tolist()}, "
            f"actual={actual_contact.tolist()}."
        )


    # ========================================================
    # INITIAL LIFTOFF MEMORY
    #
    # This is useful only when a test / restart begins
    # directly inside SS.
    #
    # Normal walking should latch liftoff through LO events.
    # ========================================================

    W_p_liftoff_L = None

    W_p_liftoff_R = None


    if initial_phase == "LSS":

        if W_p_R6 is not None:

            W_p_liftoff_R = (
                W_p_R6.copy()
            )


    elif initial_phase == "RSS":

        if W_p_L6 is not None:

            W_p_liftoff_L = (
                W_p_L6.copy()
            )


    return GaitManagerState(

        current_phase=(
            initial_phase
        ),

        t_phase=(
            t_phase
        ),

        previous_c_L=(
            c_L
        ),

        previous_c_R=(
            c_R
        ),

        W_p_liftoff_L=(
            W_p_liftoff_L
        ),

        W_p_liftoff_R=(
            W_p_liftoff_R
        ),
    )


# ============================================================
# UPDATE GAIT MANAGER
# ============================================================

def update_gait_manager(
    state,
    c_L,
    c_R,
    dt_elapsed,
    timing,
    W_p_L6=None,
    W_p_R6=None,
    W_p_foothold_L=None,
    W_p_foothold_R=None,
):
    """
    Update synchronized gait state by one control/simulation
    sample.

    Processing order
    ----------------

        previous contacts

            ->

        detect physical contact events

            ->

        advance t_phase by dt_elapsed

            ->

        synchronize FSM using expected physical event only

            ->

        latch physical liftoff position if LO occurred

            ->

        determine current swing foot

            ->

        compute swing_phase

            ->

        build nominal MPC phase preview


    Important
    ---------

    Nominal time NEVER forces an actual gait transition.

    Example:

        current_phase = LSS

    If:

        t_phase > T_SS

    but:

        TD_R has not occurred

    then:

        current_phase remains LSS

    and:

        swing_phase = 1.0

    until the actual right-foot touchdown is detected.


    This function does NOT call:

        mujoco.mj_step()

    and does NOT generate footholds.
    """

    if not isinstance(
        state,
        GaitManagerState,
    ):

        raise TypeError(
            "state phải là GaitManagerState."
        )


    # ========================================================
    # 1. CURRENT CONTACT
    # ========================================================

    c_L = normalize_contact_state(
        c_L
    )


    c_R = normalize_contact_state(
        c_R
    )


    # ========================================================
    # 2. CURRENT FOOT POSITIONS
    # ========================================================

    W_p_L6 = _validate_optional_vector3(
        W_p_L6,
        "W_p_L6"
    )


    W_p_R6 = _validate_optional_vector3(
        W_p_R6,
        "W_p_R6"
    )


    # ========================================================
    # 3. CONTACT EVENTS
    # ========================================================

    events = detect_contact_events(

        previous_left_contact=(
            state.previous_c_L
        ),

        previous_right_contact=(
            state.previous_c_R
        ),

        current_left_contact=(
            c_L
        ),

        current_right_contact=(
            c_R
        ),
    )


    # ========================================================
    # 4. ADVANCE ACTUAL PHASE TIMER
    #
    # x_k represents the current contact sample.
    #
    # dt_elapsed is elapsed time since previous gait-manager
    # update.
    # ========================================================

    t_phase_advanced = (
        advance_phase_timer(

            t_phase=(
                state.t_phase
            ),

            dt_elapsed=(
                dt_elapsed
            ),
        )
    )


    # ========================================================
    # 5. SYNCHRONIZE ACTUAL FSM
    #
    # Actual transition occurs ONLY if the event expected
    # by current_phase is present.
    # ========================================================

    (
        phase_new,
        t_phase_new,
        transitioned,
        transition_event,

    ) = synchronize_gait_phase(

        current_phase=(
            state.current_phase
        ),

        t_phase=(
            t_phase_advanced
        ),

        events=(
            events
        ),
    )


    # ========================================================
    # 6. COPY EXISTING LIFTOFF MEMORY
    # ========================================================

    if state.W_p_liftoff_L is None:

        W_p_liftoff_L = None

    else:

        W_p_liftoff_L = (
            state.W_p_liftoff_L.copy()
        )


    if state.W_p_liftoff_R is None:

        W_p_liftoff_R = None

    else:

        W_p_liftoff_R = (
            state.W_p_liftoff_R.copy()
        )


    # ========================================================
    # 7. LATCH LIFTOFF POSITION
    #
    # DS_L_TO_R -- LO_L --> RSS
    #
    #     Left foot begins swing.
    #
    #
    # DS_R_TO_L -- LO_R --> LSS
    #
    #     Right foot begins swing.
    # ========================================================

    if transition_event == "LO_L":

        if W_p_L6 is None:

            raise ValueError(
                "LO_L detected nhưng W_p_L6=None. "
                "Không thể latch left liftoff position."
            )


        W_p_liftoff_L = (
            W_p_L6.copy()
        )


    elif transition_event == "LO_R":

        if W_p_R6 is None:

            raise ValueError(
                "LO_R detected nhưng W_p_R6=None. "
                "Không thể latch right liftoff position."
            )


        W_p_liftoff_R = (
            W_p_R6.copy()
        )


    # ========================================================
    # 8. NEW PERSISTENT STATE
    # ========================================================

    state_new = GaitManagerState(

        current_phase=(
            phase_new
        ),

        t_phase=(
            float(
                t_phase_new
            )
        ),

        previous_c_L=(
            c_L
        ),

        previous_c_R=(
            c_R
        ),

        W_p_liftoff_L=(
            W_p_liftoff_L
        ),

        W_p_liftoff_R=(
            W_p_liftoff_R
        ),
    )


    # ========================================================
    # 9. PHYSICAL CONTACT MODE
    # ========================================================

    contact_mode = get_contact_mode(
        c_L,
        c_R,
    )


    # ========================================================
    # 10. PHASE / CONTACT CONSISTENCY
    # ========================================================

    expected_contact = (
        get_phase_contact_state(
            phase_new
        )
    )


    actual_contact = np.array(
        [
            c_L,
            c_R,
        ],
        dtype=int,
    )


    phase_contact_consistent = bool(

        np.array_equal(
            expected_contact,
            actual_contact,
        )
    )


    # ========================================================
    # 11. NOMINAL PHASE OVERDUE FLAG
    #
    # IMPORTANT:
    #
    # This flag NEVER changes current_phase.
    # ========================================================

    T_phase_nominal = (
        get_nominal_phase_duration(
            phase_new,
            timing
        )
    )


    phase_overdue = bool(

        t_phase_new

        >

        T_phase_nominal
    )


    # ========================================================
    # 12. CURRENT SWING FOOT
    # ========================================================

    swing_foot = (
        get_swing_foot_from_phase(
            phase_new
        )
    )


    # ========================================================
    # 13. SWING PHASE
    # ========================================================

    swing_phase = (
        compute_swing_phase(

            current_phase=(
                phase_new
            ),

            t_phase=(
                t_phase_new
            ),

            timing=(
                timing
            ),
        )
    )


    # ========================================================
    # 14. CURRENT SWING LIFTOFF POSITION
    # ========================================================

    W_p_liftoff = (
        _get_current_swing_liftoff(

            state=(
                state_new
            ),

            swing_foot=(
                swing_foot
            ),
        )
    )


    # ========================================================
    # 15. CURRENT EXTERNAL FOOTHOLD TARGET
    # ========================================================

    W_p_foothold = (
        _get_current_swing_foothold(

            swing_foot=(
                swing_foot
            ),

            W_p_foothold_L=(
                W_p_foothold_L
            ),

            W_p_foothold_R=(
                W_p_foothold_R
            ),
        )
    )


    # ========================================================
    # 16. EXPECTED NEXT PHYSICAL EVENT
    # ========================================================

    expected_event = (
        get_expected_transition_event(
            phase_new
        )
    )


    # ========================================================
    # 17. MPC NOMINAL PHASE PREVIEW
    #
    # j = 0:
    #
    #     actual synchronized phase
    #
    # j > 0:
    #
    #     nominal future prediction
    # ========================================================

    phase_preview = (
        build_phase_preview(

            current_phase=(
                phase_new
            ),

            t_phase=(
                t_phase_new
            ),

            timing=(
                timing
            ),
        )
    )


    # ========================================================
    # 18. OUTPUT
    # ========================================================

    output = GaitManagerOutput(

        current_phase=(
            phase_new
        ),

        t_phase=(
            float(
                t_phase_new
            )
        ),

        c_L=(
            c_L
        ),

        c_R=(
            c_R
        ),

        contact_mode=(
            contact_mode
        ),

        events=(
            events.copy()
        ),

        transitioned=(
            bool(
                transitioned
            )
        ),

        transition_event=(
            transition_event
        ),

        expected_event=(
            expected_event
        ),

        swing_foot=(
            swing_foot
        ),

        swing_phase=(
            swing_phase
        ),

        W_p_liftoff=(
            None
            if W_p_liftoff is None
            else W_p_liftoff.copy()
        ),

        W_p_foothold=(
            None
            if W_p_foothold is None
            else W_p_foothold.copy()
        ),

        phase_overdue=(
            phase_overdue
        ),

        phase_contact_consistent=(
            phase_contact_consistent
        ),

        phase_preview=(
            phase_preview
        ),
    )


    return (
        state_new,
        output,
    )