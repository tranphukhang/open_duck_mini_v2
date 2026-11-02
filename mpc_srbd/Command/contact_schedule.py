from __future__ import annotations

import numpy as np


# ============================================================
# PHASE DEFINITIONS
# ============================================================

PHASE_LSS = "LSS"
PHASE_DS_L_TO_R = "DS_L_TO_R"
PHASE_RSS = "RSS"
PHASE_DS_R_TO_L = "DS_R_TO_L"

VALID_PHASES = (
    PHASE_LSS,
    PHASE_DS_L_TO_R,
    PHASE_RSS,
    PHASE_DS_R_TO_L,
)

# Contact convention:
#     [c_L, c_R]
#
# LSS:
#     left stance, right swing
#
# RSS:
#     right stance, left swing
#
# Directional DS phases are kept separate because DCM / foot-plan
# logic needs to know the direction of support transfer even though
# their binary contact state is the same [1, 1].
PHASE_TO_CONTACT = {
    PHASE_LSS: np.array([1, 0], dtype=int),
    PHASE_DS_L_TO_R: np.array([1, 1], dtype=int),
    PHASE_RSS: np.array([0, 1], dtype=int),
    PHASE_DS_R_TO_L: np.array([1, 1], dtype=int),
}

# Nominal cyclic topology.
NEXT_PHASE = {
    PHASE_LSS: PHASE_DS_L_TO_R,
    PHASE_DS_L_TO_R: PHASE_RSS,
    PHASE_RSS: PHASE_DS_R_TO_L,
    PHASE_DS_R_TO_L: PHASE_LSS,
}


# ============================================================
# VALIDATION / BASIC MAPPING
# ============================================================

def _validate_phase(phase: str) -> str:
    phase = str(phase)

    if phase not in VALID_PHASES:
        raise ValueError(
            f"Unsupported gait phase: {phase!r}. "
            f"Expected one of {VALID_PHASES}."
        )

    return phase


def _validate_positive_scalar(value, name: str) -> float:
    value = float(value)

    if not np.isfinite(value):
        raise ValueError(f"{name} phải hữu hạn.")

    if value <= 0.0:
        raise ValueError(f"{name} phải > 0.")

    return value


def _validate_nonnegative_scalar(value, name: str) -> float:
    value = float(value)

    if not np.isfinite(value):
        raise ValueError(f"{name} phải hữu hạn.")

    if value < 0.0:
        raise ValueError(f"{name} phải >= 0.")

    return value


def get_phase_contact(phase: str) -> np.ndarray:
    """
    Return binary contact state [c_L, c_R] for one gait phase.

    This function is purely nominal. It does not read physical contact.
    """
    phase = _validate_phase(phase)

    return PHASE_TO_CONTACT[phase].copy()


def get_next_phase(phase: str) -> str:
    """
    Return the nominal cyclic successor of one gait phase.
    """
    phase = _validate_phase(phase)

    return NEXT_PHASE[phase]


def get_phase_duration(
    phase: str,
    timing: dict,
) -> float:
    """
    Return nominal duration of one phase.

    Required timing entries:
        timing["T_SS"]
        timing["T_DS"]
    """
    phase = _validate_phase(phase)

    if "T_SS" not in timing:
        raise KeyError("timing thiếu khóa 'T_SS'.")

    if "T_DS" not in timing:
        raise KeyError("timing thiếu khóa 'T_DS'.")

    T_SS = _validate_positive_scalar(
        timing["T_SS"],
        "timing['T_SS']",
    )

    T_DS = _validate_positive_scalar(
        timing["T_DS"],
        "timing['T_DS']",
    )

    if phase in (PHASE_LSS, PHASE_RSS):
        return T_SS

    return T_DS


# ============================================================
# NOMINAL PHASE PROPAGATION
# ============================================================

def normalize_phase_time(
    phase: str,
    t_phase: float,
    timing: dict,
):
    """
    Normalize a nominal phase/time pair.

    If t_phase is longer than the nominal duration, nominal preview
    is advanced through as many phase boundaries as necessary.

    IMPORTANT:
        This is ONLY prediction-horizon propagation.

        It does not update the physical FSM and does not consume
        contact feedback. The real current phase must already have
        been accepted by gait_manager/FSM before calling this module.
    """
    phase = _validate_phase(phase)
    t_phase = _validate_nonnegative_scalar(
        t_phase,
        "t_phase",
    )

    transitions = []

    # A small tolerance avoids floating-point boundary chatter.
    eps = 1e-12

    while True:
        duration = get_phase_duration(
            phase,
            timing,
        )

        if t_phase < duration - eps:
            break

        t_phase -= duration

        if abs(t_phase) <= eps:
            t_phase = 0.0

        previous_phase = phase
        phase = get_next_phase(
            phase
        )

        transitions.append(
            (previous_phase, phase)
        )

    return phase, float(t_phase), transitions


def advance_nominal_phase(
    phase: str,
    t_phase: float,
    delta_t: float,
    timing: dict,
):
    """
    Propagate a nominal phase/time pair by delta_t.

    This function is used only to construct the FUTURE preview.
    It must never be used to force the physical FSM to transition.
    """
    phase = _validate_phase(phase)
    t_phase = _validate_nonnegative_scalar(
        t_phase,
        "t_phase",
    )
    delta_t = _validate_nonnegative_scalar(
        delta_t,
        "delta_t",
    )

    return normalize_phase_time(
        phase=phase,
        t_phase=t_phase + delta_t,
        timing=timing,
    )


# ============================================================
# UNIFIED MPC PREVIEW
# ============================================================

def build_contact_preview(
    current_phase: str,
    t_phase: float,
    timing: dict,
    horizon: int,
    dt_mpc: float,
):
    """
    Build ONE synchronized nominal preview for the MPC horizon.

    Parameters
    ----------
    current_phase
        Physical/FSM phase already accepted at the current 100-Hz
        control tick.

    t_phase
        Current progress inside current_phase.

    timing
        Must contain:
            T_SS
            T_DS

    horizon
        MPC horizon length N.

    dt_mpc
        Prediction-grid spacing.

    Returns
    -------
    dict containing:

        phase_sequence
            List of N directional gait phases.

        phase_time_sequence
            Time inside each predicted phase.

        phase_progress_sequence
            Normalized phase progress in [0, 1).

        contact_schedule
            Array shape (N, 2), each row [c_L, c_R].

        prediction_offsets
            Interval-start offsets:
                [0, dt_mpc, ..., (N-1) dt_mpc]

    Time convention
    ---------------
    This module uses INTERVAL-START support/contact semantics.

    Therefore:
        preview[0]
    always corresponds to the current physical/FSM phase and is the
    support mode used by the current MPC input u_0.

    For example, with:
        T_SS = 0.4 s
        T_DS = 0.1 s
        dt_mpc = 0.05 s
        horizon = 10
        current_phase = LSS
        t_phase = 0

    the preview is exactly:
        8 samples LSS
        2 samples DS_L_TO_R

    which matches:
        N_SS = 8
        N_DS = 2

    IMPORTANT ARCHITECTURE RULE
    ---------------------------
    This file does NOT read:
        c_L, c_R, TD, LO

    Contact feedback belongs to contact_switch / gait_manager.
    After the FSM accepts a real transition, call this function again
    with the NEW current_phase. That rebuilds the receding horizon.
    """
    current_phase = _validate_phase(
        current_phase
    )

    t_phase = _validate_nonnegative_scalar(
        t_phase,
        "t_phase",
    )

    dt_mpc = _validate_positive_scalar(
        dt_mpc,
        "dt_mpc",
    )

    horizon = int(horizon)

    if horizon <= 0:
        raise ValueError("horizon phải là số nguyên > 0.")

    # The physical FSM is the authority for current_phase.
    #
    # We intentionally do NOT normalize current_phase/t_phase here
    # before the first sample, because doing so could silently force
    # the current support mode to a nominal successor.
    #
    # If the real FSM is waiting at a phase boundary for contact
    # confirmation, t_phase should be clamped/held by gait_manager.
    phase = current_phase
    phase_time = float(t_phase)

    duration_current = get_phase_duration(
        phase,
        timing,
    )

    if phase_time > duration_current + 1e-9:
        raise ValueError(
            "t_phase vượt quá nominal phase duration. "
            "Physical FSM/gait_manager phải hold tại boundary "
            "thay vì để contact_schedule tự chuyển current phase. "
            f"phase={phase}, t_phase={phase_time:.9f}, "
            f"duration={duration_current:.9f}"
        )

    # Permit exact boundary state while waiting for physical feedback.
    if phase_time > duration_current:
        phase_time = duration_current

    phase_sequence = []
    phase_time_sequence = np.zeros(
        horizon,
        dtype=float,
    )
    phase_progress_sequence = np.zeros(
        horizon,
        dtype=float,
    )
    contact_schedule = np.zeros(
        (horizon, 2),
        dtype=int,
    )
    prediction_offsets = (
        np.arange(
            horizon,
            dtype=float,
        )
        * dt_mpc
    )

    # Build each prediction sample directly from:
    #
    #     current accepted FSM state + j * dt_mpc
    #
    # rather than mutating the old preview array.
    for j in range(horizon):
        if j == 0:
            predicted_phase = phase
            predicted_t_phase = phase_time

        else:
            (
                predicted_phase,
                predicted_t_phase,
                _,
            ) = advance_nominal_phase(
                phase=phase,
                t_phase=phase_time,
                delta_t=prediction_offsets[j],
                timing=timing,
            )

        phase_sequence.append(
            predicted_phase
        )

        phase_time_sequence[j] = (
            predicted_t_phase
        )

        duration = get_phase_duration(
            predicted_phase,
            timing,
        )

        phase_progress_sequence[j] = float(
            np.clip(
                predicted_t_phase / duration,
                0.0,
                1.0,
            )
        )

        contact_schedule[j, :] = (
            get_phase_contact(
                predicted_phase
            )
        )

    return {
        "current_phase": current_phase,
        "current_t_phase": float(t_phase),
        "phase_sequence": phase_sequence,
        "phase_time_sequence": phase_time_sequence,
        "phase_progress_sequence": phase_progress_sequence,
        "contact_schedule": contact_schedule,
        "prediction_offsets": prediction_offsets,
    }


# ============================================================
# BACKWARD-COMPATIBLE API
# ============================================================

def build_phase_preview(
    current_phase: str,
    t_phase: float,
    timing: dict,
    horizon: int,
    dt_mpc: float,
):
    """
    Backward-compatible wrapper.

    New code should preferably use build_contact_preview() so phase
    and contact schedule always come from one returned object.
    """
    return build_contact_preview(
        current_phase=current_phase,
        t_phase=t_phase,
        timing=timing,
        horizon=horizon,
        dt_mpc=dt_mpc,
    )


def build_contact_schedule(
    current_phase: str,
    t_phase: float,
    timing: dict,
    horizon: int,
    dt_mpc: float,
):
    """
    Backward-compatible wrapper returning only the contact schedule.

    Internally this uses exactly the same preview generator as
    build_phase_preview().
    """
    preview = build_contact_preview(
        current_phase=current_phase,
        t_phase=t_phase,
        timing=timing,
        horizon=horizon,
        dt_mpc=dt_mpc,
    )

    return preview[
        "contact_schedule"
    ].copy()
