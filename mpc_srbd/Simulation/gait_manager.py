from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from Command.contact_schedule import (
    PHASE_LSS,
    PHASE_DS_L_TO_R,
    PHASE_RSS,
    PHASE_DS_R_TO_L,
    VALID_PHASES,
    get_next_phase,
    get_phase_contact,
    get_phase_duration,
)


# ============================================================
# EVENT / PHASE DEFINITIONS
# ============================================================

EVENT_TD_L = "TD_L"
EVENT_LO_L = "LO_L"
EVENT_TD_R = "TD_R"
EVENT_LO_R = "LO_R"

PHASE_EXPECTED_EVENT = {
    PHASE_LSS: EVENT_TD_R,
    PHASE_DS_L_TO_R: EVENT_LO_L,
    PHASE_RSS: EVENT_TD_L,
    PHASE_DS_R_TO_L: EVENT_LO_R,
}

PHASE_SWING_FOOT = {
    PHASE_LSS: "R",
    PHASE_DS_L_TO_R: None,
    PHASE_RSS: "L",
    PHASE_DS_R_TO_L: None,
}


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class GaitManagerState:
    """
    Persistent physical gait/FSM state.

    IMPORTANT RATE ARCHITECTURE
    ---------------------------
    update_gait_manager() is intended to be called at the HIGH-LEVEL
    control rate (100 Hz in the current project), not at MuJoCo's
    2000-Hz physics rate.

    current_phase
        Physical phase accepted by the FSM.

    t_phase
        Phase cursor. It advances with high-level control time and is
        clamped at the nominal phase boundary while waiting for contact
        feedback to authorize the next phase.

    c_L_prev, c_R_prev
        Previous 100-Hz sampled contact state. These are used only for
        TD/LO diagnostics.

    W_p_liftoff_L/R
        Realized liftoff positions latched when DS -> SS is accepted.

    W_p_foothold_L/R
        Current swing targets latched from the support-plan input.
    """
    current_phase: str
    t_phase: float

    c_L_prev: int
    c_R_prev: int

    W_p_liftoff_L: Optional[np.ndarray] = None
    W_p_liftoff_R: Optional[np.ndarray] = None

    W_p_foothold_L: Optional[np.ndarray] = None
    W_p_foothold_R: Optional[np.ndarray] = None


@dataclass
class GaitManagerOutput:
    """
    Read-only snapshot returned by update_gait_manager().
    """
    current_phase: str
    t_phase: float

    transitioned: bool
    transition_event: Optional[str]
    expected_event: str

    events: Tuple[str, ...]

    c_L: int
    c_R: int

    phase_contact_consistent: bool
    contact_matches_current_phase: bool
    contact_matches_next_phase: bool

    boundary_reached: bool
    waiting_for_contact: bool

    swing_foot: Optional[str]
    swing_phase: Optional[float]

    W_p_liftoff: Optional[np.ndarray]
    W_p_foothold: Optional[np.ndarray]


# ============================================================
# BASIC VALIDATION
# ============================================================

def _validate_phase(phase: str) -> str:
    phase = str(phase)

    if phase not in VALID_PHASES:
        raise ValueError(
            f"Unsupported gait phase: {phase!r}. "
            f"Expected one of {VALID_PHASES}."
        )

    return phase


def _validate_contact(value, name: str) -> int:
    value = int(value)

    if value not in (0, 1):
        raise ValueError(
            f"{name} phải là 0 hoặc 1."
        )

    return value


def _validate_nonnegative_scalar(value, name: str) -> float:
    value = float(value)

    if not np.isfinite(value):
        raise ValueError(
            f"{name} phải hữu hạn."
        )

    if value < 0.0:
        raise ValueError(
            f"{name} phải >= 0."
        )

    return value


def _copy_optional_vector(value):
    if value is None:
        return None

    vector = np.asarray(
        value,
        dtype=float,
    ).reshape(3).copy()

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            "Foot position/foothold chứa NaN hoặc Inf."
        )

    return vector


# ============================================================
# CONTACT EVENT DETECTION
# ============================================================

def detect_contact_events(
    c_L_prev: int,
    c_R_prev: int,
    c_L: int,
    c_R: int,
) -> Tuple[str, ...]:
    """
    Detect TD/LO edges between TWO HIGH-LEVEL contact samples.

    This is diagnostic/event information only.

    Phase transition authorization does not require that the edge occur
    exactly at the boundary sample. Instead, at the boundary the current
    physical contact state is compared with the nominal next phase.

    This is important because an expected TD/LO can occur slightly before
    the nominal boundary and remain physically valid until the boundary.
    """
    c_L_prev = _validate_contact(
        c_L_prev,
        "c_L_prev",
    )
    c_R_prev = _validate_contact(
        c_R_prev,
        "c_R_prev",
    )
    c_L = _validate_contact(
        c_L,
        "c_L",
    )
    c_R = _validate_contact(
        c_R,
        "c_R",
    )

    events = []

    if c_L_prev == 0 and c_L == 1:
        events.append(
            EVENT_TD_L
        )
    elif c_L_prev == 1 and c_L == 0:
        events.append(
            EVENT_LO_L
        )

    if c_R_prev == 0 and c_R == 1:
        events.append(
            EVENT_TD_R
        )
    elif c_R_prev == 1 and c_R == 0:
        events.append(
            EVENT_LO_R
        )

    return tuple(events)


# ============================================================
# PHYSICAL CONTACT / FSM RELATION
# ============================================================

def _contact_vector(
    c_L: int,
    c_R: int,
) -> np.ndarray:
    return np.array(
        [
            _validate_contact(c_L, "c_L"),
            _validate_contact(c_R, "c_R"),
        ],
        dtype=int,
    )


def contact_matches_phase(
    phase: str,
    c_L: int,
    c_R: int,
) -> bool:
    phase = _validate_phase(
        phase
    )

    return bool(
        np.array_equal(
            _contact_vector(c_L, c_R),
            get_phase_contact(phase),
        )
    )


def contact_is_admissible_for_current_or_next_phase(
    phase: str,
    c_L: int,
    c_R: int,
) -> bool:
    """
    Contact may match either:

        - current physical phase, or
        - nominal next phase.

    Why allow next phase before the boundary?
    -----------------------------------------
    Contact feedback is allowed to become physically ready slightly before
    the nominal phase cursor reaches its boundary.

    Example:
        current FSM = LSS  -> [1,0]
        right foot touches early -> [1,1]

    The FSM must NOT transition early, but [1,1] is still an admissible
    physical contact because it is exactly the contact state of the
    expected next phase DS_L_TO_R.
    """
    current_match = contact_matches_phase(
        phase,
        c_L,
        c_R,
    )

    next_phase = get_next_phase(
        phase
    )

    next_match = contact_matches_phase(
        next_phase,
        c_L,
        c_R,
    )

    return bool(
        current_match or next_match
    )


# ============================================================
# SWING OUTPUT HELPERS
# ============================================================

def _get_swing_phase(
    phase: str,
    t_phase: float,
    timing: dict,
) -> Optional[float]:
    swing_foot = PHASE_SWING_FOOT[
        phase
    ]

    if swing_foot is None:
        return None

    duration = get_phase_duration(
        phase,
        timing,
    )

    return float(
        np.clip(
            float(t_phase) / duration,
            0.0,
            1.0,
        )
    )


def _get_active_liftoff(
    state: GaitManagerState,
):
    if state.current_phase == PHASE_LSS:
        return _copy_optional_vector(
            state.W_p_liftoff_R
        )

    if state.current_phase == PHASE_RSS:
        return _copy_optional_vector(
            state.W_p_liftoff_L
        )

    return None


def _get_active_foothold(
    state: GaitManagerState,
):
    if state.current_phase == PHASE_LSS:
        return _copy_optional_vector(
            state.W_p_foothold_R
        )

    if state.current_phase == PHASE_RSS:
        return _copy_optional_vector(
            state.W_p_foothold_L
        )

    return None


def _latch_supplied_footholds(
    state: GaitManagerState,
    W_p_foothold_L,
    W_p_foothold_R,
):
    """
    Store whichever swing target is supplied by the higher-level support plan.

    This keeps compatibility with the current run_walking.py API.
    """
    if W_p_foothold_L is not None:
        state.W_p_foothold_L = (
            _copy_optional_vector(
                W_p_foothold_L
            )
        )

    if W_p_foothold_R is not None:
        state.W_p_foothold_R = (
            _copy_optional_vector(
                W_p_foothold_R
            )
        )


# ============================================================
# INITIALIZATION
# ============================================================

def initialize_gait_manager(
    initial_phase: str,
    c_L: int,
    c_R: int,
    W_p_L6,
    W_p_R6,
    t_phase: float = 0.0,
    strict_contact: bool = True,
):
    """
    Initialize the physical gait FSM.

    The initial phase must come from the startup/handoff logic.

    If strict_contact=True, the initial sampled contact must exactly match
    the requested physical phase.
    """
    initial_phase = _validate_phase(
        initial_phase
    )

    c_L = _validate_contact(
        c_L,
        "c_L",
    )
    c_R = _validate_contact(
        c_R,
        "c_R",
    )

    t_phase = _validate_nonnegative_scalar(
        t_phase,
        "t_phase",
    )

    phase_duration = get_phase_duration(
        initial_phase,
        {
            # This placeholder is not used here for the actual value.
            # t_phase is bounded later in update_gait_manager(), where
            # the real timing object is available.
            "T_SS": max(t_phase, 1.0),
            "T_DS": max(t_phase, 1.0),
        },
    )

    del phase_duration

    # Validate physical foot positions at initialization.
    _copy_optional_vector(
        W_p_L6
    )
    _copy_optional_vector(
        W_p_R6
    )

    if strict_contact and not contact_matches_phase(
        initial_phase,
        c_L,
        c_R,
    ):
        raise RuntimeError(
            "Initial gait/contact mismatch: "
            f"phase={initial_phase}, "
            f"contact=[{c_L},{c_R}], "
            f"expected={get_phase_contact(initial_phase).tolist()}."
        )

    state = GaitManagerState(
        current_phase=initial_phase,
        t_phase=float(t_phase),
        c_L_prev=c_L,
        c_R_prev=c_R,
    )

    return state


# ============================================================
# PHASE TRANSITION
# ============================================================

def _apply_phase_transition(
    state: GaitManagerState,
    next_phase: str,
    W_p_L6,
    W_p_R6,
):
    """
    Apply exactly ONE accepted physical phase transition.

    DS_L_TO_R -> RSS:
        left foot physically lifts -> latch L liftoff.

    DS_R_TO_L -> LSS:
        right foot physically lifts -> latch R liftoff.

    SS -> DS touchdown transitions do not create a new liftoff latch.
    """
    previous_phase = state.current_phase
    next_phase = _validate_phase(
        next_phase
    )

    if get_next_phase(previous_phase) != next_phase:
        raise RuntimeError(
            "Illegal gait transition: "
            f"{previous_phase} -> {next_phase}."
        )

    if (
        previous_phase == PHASE_DS_L_TO_R
        and next_phase == PHASE_RSS
    ):
        state.W_p_liftoff_L = (
            _copy_optional_vector(
                W_p_L6
            )
        )

    elif (
        previous_phase == PHASE_DS_R_TO_L
        and next_phase == PHASE_LSS
    ):
        state.W_p_liftoff_R = (
            _copy_optional_vector(
                W_p_R6
            )
        )

    state.current_phase = next_phase


# ============================================================
# 100-HZ FSM UPDATE
# ============================================================

def update_gait_manager(
    state: GaitManagerState,
    c_L: int,
    c_R: int,
    dt_elapsed: float,
    timing: dict,
    W_p_L6,
    W_p_R6,
    W_p_foothold_L=None,
    W_p_foothold_R=None,
):
    """
    Update the contact-synchronized gait FSM ONCE per high-level control tick.

    Intended project rate
    ---------------------
        100 Hz  -> dt_elapsed ~= 0.01 s

    MuJoCo physics and low-level joint PD/PID may run at 2000 Hz, but this
    function should not be called at the physics rate.

    Core algorithm
    --------------
    1. Sample current physical contact.
    2. Detect 100-Hz TD/LO edges for diagnostics.
    3. Advance the phase cursor toward its nominal boundary.
    4. NEVER transition before the boundary.
    5. At the boundary:
           if physical contact matches the nominal next phase:
               accept exactly one FSM transition,
               rebuild from the new physical phase on the next call.
           else:
               hold t_phase at the boundary.
    6. Contact may become ready before the boundary, but it only gives
       permission for the cursor to slide when the boundary is reached.

    This is the architecture:

        contact feedback
              |
              v
        permission / synchronization
              |
              v
        FSM phase cursor
              |
              v
        contact_schedule / DCM / MPC preview
    """
    if not isinstance(
        state,
        GaitManagerState,
    ):
        raise TypeError(
            "state phải là GaitManagerState."
        )

    c_L = _validate_contact(
        c_L,
        "c_L",
    )
    c_R = _validate_contact(
        c_R,
        "c_R",
    )

    dt_elapsed = _validate_nonnegative_scalar(
        dt_elapsed,
        "dt_elapsed",
    )

    W_p_L6 = _copy_optional_vector(
        W_p_L6
    )
    W_p_R6 = _copy_optional_vector(
        W_p_R6
    )

    _latch_supplied_footholds(
        state=state,
        W_p_foothold_L=W_p_foothold_L,
        W_p_foothold_R=W_p_foothold_R,
    )

    previous_phase = state.current_phase

    events = detect_contact_events(
        c_L_prev=state.c_L_prev,
        c_R_prev=state.c_R_prev,
        c_L=c_L,
        c_R=c_R,
    )

    duration = get_phase_duration(
        previous_phase,
        timing,
    )

    old_t_phase = float(
        state.t_phase
    )

    if old_t_phase > duration + 1e-9:
        raise RuntimeError(
            "FSM t_phase vượt quá nominal duration. "
            "Boundary phải được hold. "
            f"phase={previous_phase}, "
            f"t_phase={old_t_phase:.9f}, "
            f"duration={duration:.9f}"
        )

    old_t_phase = min(
        old_t_phase,
        duration,
    )

    candidate_t_phase = (
        old_t_phase
        +
        dt_elapsed
    )

    boundary_reached = bool(
        candidate_t_phase
        >= duration - 1e-12
    )

    # Overflow is retained ONLY if this control interval newly crosses
    # the nominal boundary and contact is already ready.
    #
    # If the cursor was already held at the boundary from a previous
    # control tick, transition happens "now", so the new phase starts at 0.
    newly_crossed_boundary = bool(
        old_t_phase < duration - 1e-12
        and boundary_reached
    )

    overflow = (
        max(
            candidate_t_phase - duration,
            0.0,
        )
        if newly_crossed_boundary
        else 0.0
    )

    next_phase = get_next_phase(
        previous_phase
    )

    contact_matches_current = (
        contact_matches_phase(
            previous_phase,
            c_L,
            c_R,
        )
    )

    contact_matches_next = (
        contact_matches_phase(
            next_phase,
            c_L,
            c_R,
        )
    )

    transitioned = False
    transition_event = None

    if not boundary_reached:
        # Cursor slides normally.
        state.t_phase = float(
            candidate_t_phase
        )

    else:
        # Cursor reaches nominal end and is clamped there unless the
        # physical contact already authorizes the nominal successor.
        state.t_phase = float(
            duration
        )

        if contact_matches_next:
            transitioned = True

            transition_event = (
                PHASE_EXPECTED_EVENT[
                    previous_phase
                ]
            )

            _apply_phase_transition(
                state=state,
                next_phase=next_phase,
                W_p_L6=W_p_L6,
                W_p_R6=W_p_R6,
            )

            new_duration = get_phase_duration(
                state.current_phase,
                timing,
            )

            state.t_phase = float(
                min(
                    overflow,
                    new_duration,
                )
            )

    # Update previous sampled contact ONLY after all logic for this
    # 100-Hz sample has been evaluated.
    state.c_L_prev = c_L
    state.c_R_prev = c_R

    current_phase = state.current_phase

    current_match_after_transition = (
        contact_matches_phase(
            current_phase,
            c_L,
            c_R,
        )
    )

    next_phase_after_transition = (
        get_next_phase(
            current_phase
        )
    )

    next_match_after_transition = (
        contact_matches_phase(
            next_phase_after_transition,
            c_L,
            c_R,
        )
    )

    phase_contact_consistent = bool(
        current_match_after_transition
        or next_match_after_transition
    )

    current_duration = get_phase_duration(
        current_phase,
        timing,
    )

    current_boundary_reached = bool(
        state.t_phase
        >= current_duration - 1e-12
    )

    waiting_for_contact = bool(
        current_boundary_reached
        and not next_match_after_transition
    )

    swing_foot = (
        PHASE_SWING_FOOT[
            current_phase
        ]
    )

    swing_phase = _get_swing_phase(
        phase=current_phase,
        t_phase=state.t_phase,
        timing=timing,
    )

    output = GaitManagerOutput(
        current_phase=current_phase,
        t_phase=float(
            state.t_phase
        ),
        transitioned=bool(
            transitioned
        ),
        transition_event=transition_event,
        expected_event=(
            PHASE_EXPECTED_EVENT[
                current_phase
            ]
        ),
        events=events,
        c_L=c_L,
        c_R=c_R,
        phase_contact_consistent=(
            phase_contact_consistent
        ),
        contact_matches_current_phase=(
            current_match_after_transition
        ),
        contact_matches_next_phase=(
            next_match_after_transition
        ),
        boundary_reached=(
            current_boundary_reached
        ),
        waiting_for_contact=(
            waiting_for_contact
        ),
        swing_foot=swing_foot,
        swing_phase=swing_phase,
        W_p_liftoff=(
            _get_active_liftoff(
                state
            )
        ),
        W_p_foothold=(
            _get_active_foothold(
                state
            )
        ),
    )

    return state, output
