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


EVENT_TD_L = "TD_L"
EVENT_LO_L = "LO_L"
EVENT_TD_R = "TD_R"
EVENT_LO_R = "LO_R"

PHASE_SWING_FOOT = {
    PHASE_LSS: "R",
    PHASE_DS_L_TO_R: None,
    PHASE_RSS: "L",
    PHASE_DS_R_TO_L: None,
}


@dataclass
class GaitManagerState:
    # COMMAND state
    current_phase: str
    t_phase: float

    # measured-contact memory
    c_L_prev: int
    c_R_prev: int

    # Swing geometry latched once at DS -> SS.
    W_p_swing_start_L: Optional[np.ndarray] = None
    W_p_swing_start_R: Optional[np.ndarray] = None
    W_p_foothold_L: Optional[np.ndarray] = None
    W_p_foothold_R: Optional[np.ndarray] = None

    @property
    def phase_cmd(self) -> str:
        return self.current_phase

    @property
    def t_phase_cmd(self) -> float:
        return float(self.t_phase)


@dataclass
class GaitManagerOutput:
    current_phase: str
    t_phase: float
    phase_cmd: str
    t_phase_cmd: float

    transitioned: bool
    transition_reason: Optional[str]
    previous_phase_cmd: str

    events: Tuple[str, ...]
    c_L: int
    c_R: int
    contact_meas: np.ndarray

    contact_cmd: np.ndarray
    contact_cmd_achieved: bool

    phase_contact_consistent: bool
    contact_matches_current_phase: bool
    contact_matches_next_phase: bool

    boundary_reached: bool
    waiting_for_contact: bool

    pending_liftoff_confirmation: bool
    expected_liftoff_event: Optional[str]
    expected_touchdown_event: Optional[str]

    swing_foot: Optional[str]
    swing_phase: Optional[float]

    W_p_swing_start: Optional[np.ndarray]
    W_p_foothold: Optional[np.ndarray]

    # Backward-compatible alias.
    W_p_liftoff: Optional[np.ndarray]


# ============================================================
# VALIDATION
# ============================================================

def _validate_phase(phase: str) -> str:
    phase = str(phase)
    if phase not in VALID_PHASES:
        raise ValueError(
            f"Unsupported phase {phase!r}; expected one of {VALID_PHASES}."
        )
    return phase


def _validate_contact(value, name: str) -> int:
    value = int(value)
    if value not in (0, 1):
        raise ValueError(f"{name} phải là 0 hoặc 1.")
    return value


def _validate_nonnegative(value, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} phải hữu hạn và >= 0.")
    return value


def _vec3(value, name: str) -> Optional[np.ndarray]:
    if value is None:
        return None

    value = np.asarray(value, dtype=float).reshape(-1)
    if value.shape != (3,):
        raise ValueError(f"{name} phải có shape (3,).")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} chứa NaN/Inf.")

    return value.copy()


def _contact_vector(c_L: int, c_R: int) -> np.ndarray:
    return np.array(
        [
            _validate_contact(c_L, "c_L"),
            _validate_contact(c_R, "c_R"),
        ],
        dtype=int,
    )


# ============================================================
# MEASURED CONTACT EVENTS
# ============================================================

def detect_contact_events(
    c_L_prev: int,
    c_R_prev: int,
    c_L: int,
    c_R: int,
) -> Tuple[str, ...]:
    c_L_prev = _validate_contact(c_L_prev, "c_L_prev")
    c_R_prev = _validate_contact(c_R_prev, "c_R_prev")
    c_L = _validate_contact(c_L, "c_L")
    c_R = _validate_contact(c_R, "c_R")

    events = []

    if c_L_prev == 0 and c_L == 1:
        events.append(EVENT_TD_L)
    elif c_L_prev == 1 and c_L == 0:
        events.append(EVENT_LO_L)

    if c_R_prev == 0 and c_R == 1:
        events.append(EVENT_TD_R)
    elif c_R_prev == 1 and c_R == 0:
        events.append(EVENT_LO_R)

    return tuple(events)


def contact_matches_phase(phase: str, c_L: int, c_R: int) -> bool:
    phase = _validate_phase(phase)
    return bool(
        np.array_equal(
            _contact_vector(c_L, c_R),
            get_phase_contact(phase),
        )
    )


# ============================================================
# SWING HELPERS
# ============================================================

def _expected_liftoff_event(phase: str) -> Optional[str]:
    if phase == PHASE_LSS:
        return EVENT_LO_R
    if phase == PHASE_RSS:
        return EVENT_LO_L
    return None


def _expected_touchdown_event(phase: str) -> Optional[str]:
    if phase == PHASE_LSS:
        return EVENT_TD_R
    if phase == PHASE_RSS:
        return EVENT_TD_L
    return None


def _swing_phase(
    phase: str,
    t_phase: float,
    timing: dict,
) -> Optional[float]:
    if PHASE_SWING_FOOT[phase] is None:
        return None

    T_phase = get_phase_duration(phase, timing)
    return float(np.clip(t_phase / T_phase, 0.0, 1.0))


def _active_swing_start(
    state: GaitManagerState,
) -> Optional[np.ndarray]:
    if state.current_phase == PHASE_LSS:
        return _vec3(state.W_p_swing_start_R, "W_p_swing_start_R")
    if state.current_phase == PHASE_RSS:
        return _vec3(state.W_p_swing_start_L, "W_p_swing_start_L")
    return None


def _active_foothold(
    state: GaitManagerState,
) -> Optional[np.ndarray]:
    if state.current_phase == PHASE_LSS:
        return _vec3(state.W_p_foothold_R, "W_p_foothold_R")
    if state.current_phase == PHASE_RSS:
        return _vec3(state.W_p_foothold_L, "W_p_foothold_L")
    return None


def _update_next_swing_target_only_in_ds(
    state: GaitManagerState,
    W_p_foothold_L,
    W_p_foothold_R,
):
    """
    Update the next swing target ONLY while the command phase is DS.

    Once DS -> SS occurs, the active target is frozen for the complete
    swing.  Subsequent 100-Hz calls during SS are not allowed to move it.
    """
    if state.current_phase == PHASE_DS_L_TO_R:
        if W_p_foothold_L is not None:
            state.W_p_foothold_L = _vec3(
                W_p_foothold_L,
                "W_p_foothold_L",
            )

    elif state.current_phase == PHASE_DS_R_TO_L:
        if W_p_foothold_R is not None:
            state.W_p_foothold_R = _vec3(
                W_p_foothold_R,
                "W_p_foothold_R",
            )

    # In LSS/RSS the active swing target is intentionally not overwritten.


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
    initial_phase = _validate_phase(initial_phase)
    c_L = _validate_contact(c_L, "c_L")
    c_R = _validate_contact(c_R, "c_R")
    t_phase = _validate_nonnegative(t_phase, "t_phase")

    _vec3(W_p_L6, "W_p_L6")
    _vec3(W_p_R6, "W_p_R6")

    if strict_contact and not contact_matches_phase(
        initial_phase,
        c_L,
        c_R,
    ):
        raise RuntimeError(
            "Initial phase/contact mismatch: "
            f"phase_cmd={initial_phase}, "
            f"contact_meas=[{c_L},{c_R}], "
            f"contact_cmd={get_phase_contact(initial_phase).tolist()}."
        )

    return GaitManagerState(
        current_phase=initial_phase,
        t_phase=t_phase,
        c_L_prev=c_L,
        c_R_prev=c_R,
    )


# ============================================================
# COMMAND PHASE TRANSITION
# ============================================================

def _transition_command_phase(
    state: GaitManagerState,
    next_phase: str,
    W_p_L6: np.ndarray,
    W_p_R6: np.ndarray,
):
    """
    Apply exactly one command transition.

    At DS -> SS:
        - latch swing start from measured outgoing-foot position;
        - keep the target already stored during DS;
        - both endpoints then remain fixed for the whole SS interval.
    """
    previous_phase = state.current_phase
    next_phase = _validate_phase(next_phase)

    if get_next_phase(previous_phase) != next_phase:
        raise RuntimeError(
            f"Illegal command transition {previous_phase} -> {next_phase}."
        )

    if (
        previous_phase == PHASE_DS_L_TO_R
        and next_phase == PHASE_RSS
    ):
        if state.W_p_foothold_L is None:
            raise RuntimeError(
                "DS_L_TO_R -> RSS nhưng chưa có L swing foothold target."
            )
        state.W_p_swing_start_L = W_p_L6.copy()

    elif (
        previous_phase == PHASE_DS_R_TO_L
        and next_phase == PHASE_LSS
    ):
        if state.W_p_foothold_R is None:
            raise RuntimeError(
                "DS_R_TO_L -> LSS nhưng chưa có R swing foothold target."
            )
        state.W_p_swing_start_R = W_p_R6.copy()

    state.current_phase = next_phase


# ============================================================
# 100-HZ UPDATE
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
    Command/feedback gait update.

    Rules
    -----
    DS -> SS:
        timing-driven at T_DS.

    SS -> DS:
        touchdown-confirmed at/after T_SS.

    Swing geometry
    --------------
    During DS, the next target may be refreshed by the foot planner.
    At DS -> SS, swing start is latched.  The target stored in DS is then
    held fixed for the full swing.  Only swing_phase changes at 100 Hz.
    """
    if not isinstance(state, GaitManagerState):
        raise TypeError("state phải là GaitManagerState.")

    c_L = _validate_contact(c_L, "c_L")
    c_R = _validate_contact(c_R, "c_R")
    dt_elapsed = _validate_nonnegative(dt_elapsed, "dt_elapsed")

    W_p_L6 = _vec3(W_p_L6, "W_p_L6")
    W_p_R6 = _vec3(W_p_R6, "W_p_R6")

    # IMPORTANT: only DS is allowed to update the next swing target.
    _update_next_swing_target_only_in_ds(
        state=state,
        W_p_foothold_L=W_p_foothold_L,
        W_p_foothold_R=W_p_foothold_R,
    )

    previous_phase = state.current_phase

    events = detect_contact_events(
        state.c_L_prev,
        state.c_R_prev,
        c_L,
        c_R,
    )

    T_phase = get_phase_duration(previous_phase, timing)
    old_t = min(float(state.t_phase), T_phase)
    candidate_t = old_t + dt_elapsed

    boundary_reached = bool(candidate_t >= T_phase - 1e-12)
    newly_crossed = bool(
        old_t < T_phase - 1e-12
        and boundary_reached
    )

    overflow = (
        max(candidate_t - T_phase, 0.0)
        if newly_crossed
        else 0.0
    )

    next_phase = get_next_phase(previous_phase)
    transitioned = False
    transition_reason = None

    if not boundary_reached:
        state.t_phase = float(candidate_t)

    else:
        state.t_phase = float(T_phase)

        # ----------------------------------------------------
        # DS -> SS: timing-driven command.
        # ----------------------------------------------------
        if previous_phase in (
            PHASE_DS_L_TO_R,
            PHASE_DS_R_TO_L,
        ):
            transitioned = True
            transition_reason = "DS_TIME_COMPLETE"

            _transition_command_phase(
                state,
                next_phase,
                W_p_L6,
                W_p_R6,
            )

            T_new = get_phase_duration(state.current_phase, timing)
            state.t_phase = float(min(overflow, T_new))

        # ----------------------------------------------------
        # SS -> DS: measured touchdown required.
        # ----------------------------------------------------
        elif previous_phase in (
            PHASE_LSS,
            PHASE_RSS,
        ):
            touchdown_confirmed = contact_matches_phase(
                next_phase,
                c_L,
                c_R,
            )

            if touchdown_confirmed:
                transitioned = True
                transition_reason = "TOUCHDOWN_CONFIRMED"

                _transition_command_phase(
                    state,
                    next_phase,
                    W_p_L6,
                    W_p_R6,
                )

                T_new = get_phase_duration(state.current_phase, timing)
                state.t_phase = float(min(overflow, T_new))

            # Otherwise hold SS command at T_SS.

        else:
            raise RuntimeError(f"Unhandled phase {previous_phase}.")

    state.c_L_prev = c_L
    state.c_R_prev = c_R

    # ========================================================
    # OUTPUT
    # ========================================================
    current_phase = state.current_phase
    contact_meas = _contact_vector(c_L, c_R)
    contact_cmd = get_phase_contact(current_phase)

    contact_cmd_achieved = bool(
        np.array_equal(contact_meas, contact_cmd)
    )

    current_match = contact_matches_phase(
        current_phase,
        c_L,
        c_R,
    )

    next_phase_now = get_next_phase(current_phase)
    next_match = contact_matches_phase(
        next_phase_now,
        c_L,
        c_R,
    )

    phase_contact_consistent = bool(current_match or next_match)

    T_current = get_phase_duration(current_phase, timing)
    boundary_now = bool(state.t_phase >= T_current - 1e-12)

    waiting_for_contact = bool(
        current_phase in (PHASE_LSS, PHASE_RSS)
        and boundary_now
        and not contact_matches_phase(
            next_phase_now,
            c_L,
            c_R,
        )
    )

    swing_foot = PHASE_SWING_FOOT[current_phase]
    swing_phase = _swing_phase(
        current_phase,
        state.t_phase,
        timing,
    )

    W_p_swing_start = _active_swing_start(state)
    W_p_foothold = _active_foothold(state)

    expected_lo = _expected_liftoff_event(current_phase)
    expected_td = _expected_touchdown_event(current_phase)

    pending_liftoff_confirmation = bool(
        current_phase in (PHASE_LSS, PHASE_RSS)
        and not contact_cmd_achieved
        and np.array_equal(
            contact_meas,
            np.array([1, 1], dtype=int),
        )
    )

    output = GaitManagerOutput(
        current_phase=current_phase,
        t_phase=float(state.t_phase),
        phase_cmd=current_phase,
        t_phase_cmd=float(state.t_phase),

        transitioned=transitioned,
        transition_reason=transition_reason,
        previous_phase_cmd=previous_phase,

        events=events,
        c_L=c_L,
        c_R=c_R,
        contact_meas=contact_meas.copy(),

        contact_cmd=contact_cmd.copy(),
        contact_cmd_achieved=contact_cmd_achieved,

        phase_contact_consistent=phase_contact_consistent,
        contact_matches_current_phase=current_match,
        contact_matches_next_phase=next_match,

        boundary_reached=boundary_now,
        waiting_for_contact=waiting_for_contact,

        pending_liftoff_confirmation=pending_liftoff_confirmation,
        expected_liftoff_event=expected_lo,
        expected_touchdown_event=expected_td,

        swing_foot=swing_foot,
        swing_phase=swing_phase,
        W_p_swing_start=W_p_swing_start,
        W_p_foothold=W_p_foothold,
        W_p_liftoff=W_p_swing_start,
    )

    return state, output
