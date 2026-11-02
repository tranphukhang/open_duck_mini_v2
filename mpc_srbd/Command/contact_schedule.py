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

# Nominal / commanded contact convention: [c_L, c_R]
PHASE_TO_CONTACT = {
    PHASE_LSS: np.array([1, 0], dtype=int),
    PHASE_DS_L_TO_R: np.array([1, 1], dtype=int),
    PHASE_RSS: np.array([0, 1], dtype=int),
    PHASE_DS_R_TO_L: np.array([1, 1], dtype=int),
}

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


def _validate_contact_vector(value, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=int).reshape(-1)
    if value.shape != (2,):
        raise ValueError(f"{name} phải có shape (2,), nhận {value.shape}.")
    if not np.all(np.isin(value, [0, 1])):
        raise ValueError(f"{name} chỉ được chứa 0 hoặc 1.")
    return value.copy()


def get_phase_contact(phase: str) -> np.ndarray:
    """Return nominal/commanded binary contact [c_L, c_R]."""
    phase = _validate_phase(phase)
    return PHASE_TO_CONTACT[phase].copy()


def get_next_phase(phase: str) -> str:
    phase = _validate_phase(phase)
    return NEXT_PHASE[phase]


def get_phase_duration(phase: str, timing: dict) -> float:
    phase = _validate_phase(phase)

    if "T_SS" not in timing:
        raise KeyError("timing thiếu khóa 'T_SS'.")
    if "T_DS" not in timing:
        raise KeyError("timing thiếu khóa 'T_DS'.")

    T_SS = _validate_positive_scalar(timing["T_SS"], "timing['T_SS']")
    T_DS = _validate_positive_scalar(timing["T_DS"], "timing['T_DS']")

    if phase in (PHASE_LSS, PHASE_RSS):
        return T_SS
    return T_DS


# ============================================================
# NOMINAL COMMAND-PHASE PROPAGATION
# ============================================================

def normalize_phase_time(phase: str, t_phase: float, timing: dict):
    phase = _validate_phase(phase)
    t_phase = _validate_nonnegative_scalar(t_phase, "t_phase")

    transitions = []
    eps = 1e-12

    while True:
        duration = get_phase_duration(phase, timing)
        if t_phase < duration - eps:
            break

        t_phase -= duration
        if abs(t_phase) <= eps:
            t_phase = 0.0

        previous_phase = phase
        phase = get_next_phase(phase)
        transitions.append((previous_phase, phase))

    return phase, float(t_phase), transitions


def advance_nominal_phase(
    phase: str,
    t_phase: float,
    delta_t: float,
    timing: dict,
):
    phase = _validate_phase(phase)
    t_phase = _validate_nonnegative_scalar(t_phase, "t_phase")
    delta_t = _validate_nonnegative_scalar(delta_t, "delta_t")

    return normalize_phase_time(
        phase=phase,
        t_phase=t_phase + delta_t,
        timing=timing,
    )


# ============================================================
# COMMAND PREVIEW + MPC CONTACT PREVIEW
# ============================================================

def build_contact_preview(
    current_phase: str,
    t_phase: float,
    timing: dict,
    horizon: int,
    dt_mpc: float,
    current_contact_meas=None,
):
    """
    Build one command-phase preview and one MPC contact schedule.

    Interval-start convention
    -------------------------
    Decision variable:
        U = [u_0, u_1, ..., u_{N-1}]

    Therefore row 0 corresponds to the CURRENT input u_k applied at x_k.

    The two contact sequences are intentionally different during a
    contact transition:

        contact_schedule_cmd[j]
            Nominal/commanded contact generated from phase_cmd.

        contact_schedule_mpc[0]
            CURRENT measured physical contact, when current_contact_meas
            is supplied.

        contact_schedule_mpc[j], j >= 1
            Future commanded contact preview.

    Example immediately after DS_R_TO_L -> LSS command:

        contact_meas = [1, 1]
        contact_cmd  = [1, 0]

        contact_schedule_mpc =
            [[1, 1],     <- current physical u_k constraint
             [1, 0],     <- future desired contact
             [1, 0],
             ...]

    phase_sequence always remains the COMMAND phase sequence.  DCM and
    footstep planning can therefore use desired future phases, while the
    QP contact constraint is physically anchored at row 0.
    """
    current_phase = _validate_phase(current_phase)
    t_phase = _validate_nonnegative_scalar(t_phase, "t_phase")
    dt_mpc = _validate_positive_scalar(dt_mpc, "dt_mpc")

    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon phải là số nguyên > 0.")

    phase = current_phase
    phase_time = float(t_phase)
    duration_current = get_phase_duration(phase, timing)

    if phase_time > duration_current + 1e-9:
        raise ValueError(
            "t_phase vượt nominal phase duration. "
            "gait_manager phải hold/transition phase_cmd trước khi build preview. "
            f"phase={phase}, t_phase={phase_time:.9f}, "
            f"duration={duration_current:.9f}"
        )

    if phase_time > duration_current:
        phase_time = duration_current

    phase_sequence = []
    phase_time_sequence = np.zeros(horizon, dtype=float)
    phase_progress_sequence = np.zeros(horizon, dtype=float)
    contact_schedule_cmd = np.zeros((horizon, 2), dtype=int)
    prediction_offsets = np.arange(horizon, dtype=float) * dt_mpc

    for j in range(horizon):
        if j == 0:
            predicted_phase = phase
            predicted_t_phase = phase_time
        else:
            predicted_phase, predicted_t_phase, _ = advance_nominal_phase(
                phase=phase,
                t_phase=phase_time,
                delta_t=prediction_offsets[j],
                timing=timing,
            )

        phase_sequence.append(predicted_phase)
        phase_time_sequence[j] = predicted_t_phase

        duration = get_phase_duration(predicted_phase, timing)
        phase_progress_sequence[j] = float(
            np.clip(predicted_t_phase / duration, 0.0, 1.0)
        )

        contact_schedule_cmd[j, :] = get_phase_contact(predicted_phase)

    contact_schedule_mpc = contact_schedule_cmd.copy()

    if current_contact_meas is not None:
        current_contact_meas = _validate_contact_vector(
            current_contact_meas,
            "current_contact_meas",
        )
        contact_schedule_mpc[0, :] = current_contact_meas

    return {
        "current_phase": current_phase,
        "current_t_phase": float(t_phase),
        "current_contact_meas": (
            None
            if current_contact_meas is None
            else current_contact_meas.copy()
        ),
        "phase_sequence": phase_sequence,
        "phase_time_sequence": phase_time_sequence,
        "phase_progress_sequence": phase_progress_sequence,
        "contact_schedule_cmd": contact_schedule_cmd,
        "contact_schedule_mpc": contact_schedule_mpc,
        # Backward-compatible nominal alias.  DCM/reference code historically
        # reads this key, so keep it purely command/phase based.
        # MPC code must explicitly use "contact_schedule_mpc".
        "contact_schedule": contact_schedule_cmd.copy(),
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
    current_contact_meas=None,
):
    return build_contact_preview(
        current_phase=current_phase,
        t_phase=t_phase,
        timing=timing,
        horizon=horizon,
        dt_mpc=dt_mpc,
        current_contact_meas=current_contact_meas,
    )


def build_contact_schedule(
    current_phase: str,
    t_phase: float,
    timing: dict,
    horizon: int,
    dt_mpc: float,
    current_contact_meas=None,
):
    preview = build_contact_preview(
        current_phase=current_phase,
        t_phase=t_phase,
        timing=timing,
        horizon=horizon,
        dt_mpc=dt_mpc,
        current_contact_meas=current_contact_meas,
    )

    return preview["contact_schedule_mpc"].copy()
