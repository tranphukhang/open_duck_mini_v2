from __future__ import annotations

import numpy as np

try:
    from .contact_schedule import (
        PHASE_LSS,
        PHASE_DS_L_TO_R,
        PHASE_RSS,
        PHASE_DS_R_TO_L,
        VALID_PHASES,
        get_next_phase,
    )
except ImportError:
    from contact_schedule import (
        PHASE_LSS,
        PHASE_DS_L_TO_R,
        PHASE_RSS,
        PHASE_DS_R_TO_L,
        VALID_PHASES,
        get_next_phase,
    )


# ============================================================
# VALIDATION
# ============================================================

def _normalize_phase(phase: str) -> str:
    phase = str(phase)
    if phase not in VALID_PHASES:
        raise ValueError(
            f"Unsupported gait phase: {phase!r}. "
            f"Expected one of {VALID_PHASES}."
        )
    return phase


def _validate_position3(value, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float).reshape(-1)
    if value.shape != (3,):
        raise ValueError(
            f"{name} phải có shape (3,), nhưng nhận {value.shape}."
        )
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} chứa NaN hoặc Inf.")
    return value.copy()


def _validate_contact2(value, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=int).reshape(-1)
    if value.shape != (2,):
        raise ValueError(
            f"{name} phải có shape (2,), nhưng nhận {value.shape}."
        )
    if not np.all(np.isin(value, [0, 1])):
        raise ValueError(f"{name} chỉ được chứa 0 hoặc 1.")
    return value.copy()


def _validate_support_points(support_points) -> np.ndarray:
    support_points = np.asarray(support_points, dtype=float)

    if support_points.ndim != 2 or support_points.shape[1] != 3:
        raise ValueError("support_points phải có shape (M,3).")
    if support_points.shape[0] < 2:
        raise ValueError("support_points cần ít nhất 2 điểm.")
    if not np.all(np.isfinite(support_points)):
        raise ValueError("support_points chứa NaN hoặc Inf.")

    return support_points.copy()


def _validate_phase_sequence(phase_sequence):
    if not isinstance(phase_sequence, (list, tuple)):
        raise TypeError("phase_sequence phải là list hoặc tuple.")
    if len(phase_sequence) == 0:
        raise ValueError("phase_sequence không được rỗng.")
    return [_normalize_phase(phase) for phase in phase_sequence]


# ============================================================
# SUPPORT SIDE
# ============================================================

def get_initial_support_foot(current_phase):
    """
    Side represented by support_points[0].

    LSS / DS_L_TO_R -> L
    RSS / DS_R_TO_L -> R
    """
    current_phase = _normalize_phase(current_phase)

    if current_phase in (PHASE_LSS, PHASE_DS_L_TO_R):
        return "L"
    return "R"


def get_support_foot_from_index(initial_support_foot, support_index):
    initial_support_foot = str(initial_support_foot).upper()
    if initial_support_foot not in ("L", "R"):
        raise ValueError("initial_support_foot phải là 'L' hoặc 'R'.")

    support_index = int(support_index)
    if support_index < 0:
        raise ValueError("support_index không được âm.")

    is_even = support_index % 2 == 0

    if initial_support_foot == "L":
        return "L" if is_even else "R"
    return "R" if is_even else "L"


# ============================================================
# SUPPORT INDEX UPDATE
# ============================================================

def _is_support_advance_transition(phase_from, phase_to):
    return bool(
        (
            phase_from == PHASE_DS_L_TO_R
            and phase_to == PHASE_RSS
        )
        or
        (
            phase_from == PHASE_DS_R_TO_L
            and phase_to == PHASE_LSS
        )
    )


def advance_support_index_between_phases(
    previous_phase,
    current_phase,
    support_index,
):
    previous_phase = _normalize_phase(previous_phase)
    current_phase = _normalize_phase(current_phase)
    support_index = int(support_index)

    if support_index < 0:
        raise ValueError("support_index không được âm.")

    if previous_phase == current_phase:
        return support_index

    phase = previous_phase

    for _ in range(4):
        next_phase = get_next_phase(phase)

        if _is_support_advance_transition(phase, next_phase):
            support_index += 1

        phase = next_phase

        if phase == current_phase:
            return support_index

    raise RuntimeError(
        "Không thể map phase transition trong nominal gait cycle."
    )


# ============================================================
# FEEDBACK RE-ANCHORING
# ============================================================

def reanchor_support_points_from_feedback(
    current_phase,
    support_points,
    W_p_L6_feedback,
    W_p_R6_feedback,
):
    """
    Re-anchor realized support geometry while preserving the ACTIVE
    swing target during SS.

    Important swing-target rule
    ---------------------------
    During LSS/RSS, support_points[1] is the active touchdown target of
    the current swing foot.  It must remain fixed throughout the swing.

    Therefore, in SS:

        new_points[0] = measured stance foot
        new_points[1] = old_points[1]      # latched swing target

    and only support_points[2:] are rebuilt from the original relative
    displacement sequence.

    This prevents 100-Hz stance-foot feedback from continuously moving
    the endpoint of an already-running swing trajectory.

    During DS, both feet are physical support geometry and are therefore
    refreshed directly from measurement before rebuilding future points.
    """
    current_phase = _normalize_phase(current_phase)
    old_points = _validate_support_points(support_points)
    W_p_L6_feedback = _validate_position3(
        W_p_L6_feedback,
        "W_p_L6_feedback",
    )
    W_p_R6_feedback = _validate_position3(
        W_p_R6_feedback,
        "W_p_R6_feedback",
    )

    new_points = old_points.copy()
    delta_sequence = old_points[1:] - old_points[:-1]

    if current_phase == PHASE_LSS:
        # Current stance = L.
        new_points[0] = W_p_L6_feedback

        # Do NOT modify new_points[1]: it is the latched R touchdown target.
        start_rebuild_index = 2

    elif current_phase == PHASE_RSS:
        # Current stance = R.
        new_points[0] = W_p_R6_feedback

        # Do NOT modify new_points[1]: it is the latched L touchdown target.
        start_rebuild_index = 2

    elif current_phase == PHASE_DS_L_TO_R:
        new_points[0] = W_p_L6_feedback
        new_points[1] = W_p_R6_feedback
        start_rebuild_index = 2

    elif current_phase == PHASE_DS_R_TO_L:
        new_points[0] = W_p_R6_feedback
        new_points[1] = W_p_L6_feedback
        start_rebuild_index = 2

    else:
        raise RuntimeError(f"Unhandled phase: {current_phase}")

    for index in range(start_rebuild_index, new_points.shape[0]):
        new_points[index] = (
            new_points[index - 1]
            + delta_sequence[index - 1]
        )

    return {
        "support_points_reanchored": new_points,
        "support_points_original": old_points,
        "delta_sequence": delta_sequence,
        "initial_support_foot": get_initial_support_foot(current_phase),
    }


# ============================================================
# LEFT / RIGHT PLANNED POSITION FOR ONE PHASE
# ============================================================

def get_planned_feet_for_phase(
    phase,
    support_index,
    support_points,
    initial_support_foot,
):
    phase = _normalize_phase(phase)
    support_points = _validate_support_points(support_points)
    support_index = int(support_index)

    next_index = support_index + 1
    if next_index >= support_points.shape[0]:
        raise ValueError(
            "Không đủ support_points để xác định next touchdown foothold."
        )

    current_support_foot = get_support_foot_from_index(
        initial_support_foot,
        support_index,
    )
    next_support_foot = get_support_foot_from_index(
        initial_support_foot,
        next_index,
    )

    if current_support_foot == next_support_foot:
        raise RuntimeError("Ordered support sequence phải luân phiên L/R.")

    current_position = support_points[support_index]
    next_position = support_points[next_index]

    if current_support_foot == "L":
        W_p_L6_plan = current_position.copy()
        W_p_R6_plan = next_position.copy()
    else:
        W_p_R6_plan = current_position.copy()
        W_p_L6_plan = next_position.copy()

    if phase == PHASE_LSS and current_support_foot != "L":
        raise RuntimeError("LSS nhưng current support point không phải chân L.")
    if phase == PHASE_RSS and current_support_foot != "R":
        raise RuntimeError("RSS nhưng current support point không phải chân R.")
    if phase == PHASE_DS_L_TO_R and current_support_foot != "L":
        raise RuntimeError("DS_L_TO_R yêu cầu outgoing support là chân L.")
    if phase == PHASE_DS_R_TO_L and current_support_foot != "R":
        raise RuntimeError("DS_R_TO_L yêu cầu outgoing support là chân R.")

    return W_p_L6_plan, W_p_R6_plan


# ============================================================
# COMPLETE FOOT PLAN HORIZON
# ============================================================

def build_foot_plan_horizon(
    *,
    current_phase,
    phase_sequence,
    support_points,
    W_p_L6_feedback,
    W_p_R6_feedback,
    current_contact_meas=None,
):
    """
    Build planned L/R foot positions over the prediction horizon.

    Two distinct roles of feedback are handled here:

    1) Support-queue re-anchoring
       - DS: both realized feet anchor the queue.
       - SS: only the stance foot anchors the queue; the active swing
         target remains fixed.

    2) MPC row-0 physical geometry
       If current_contact_meas is supplied, every foot that is physically
       in contact at the current instant uses its MEASURED position in row 0.

       This is required because MPC row 0 represents u_k at x_k.  Example:

           phase_cmd    = LSS
           contact_meas = [1, 1]

       The future right-foot target may already be the commanded touchdown
       target, but row-0 right moment arm must still use the actual measured
       right-foot position because the foot is physically in contact now.
    """
    current_phase = _normalize_phase(current_phase)
    phase_sequence = _validate_phase_sequence(phase_sequence)

    if phase_sequence[0] != current_phase:
        raise ValueError("phase_sequence[0] phải bằng current_phase.")

    W_p_L6_feedback = _validate_position3(
        W_p_L6_feedback,
        "W_p_L6_feedback",
    )
    W_p_R6_feedback = _validate_position3(
        W_p_R6_feedback,
        "W_p_R6_feedback",
    )

    feedback_reanchor = reanchor_support_points_from_feedback(
        current_phase=current_phase,
        support_points=support_points,
        W_p_L6_feedback=W_p_L6_feedback,
        W_p_R6_feedback=W_p_R6_feedback,
    )

    support_points_reanchored = feedback_reanchor[
        "support_points_reanchored"
    ]

    N = len(phase_sequence)
    W_p_L6_plan = np.zeros((N, 3), dtype=float)
    W_p_R6_plan = np.zeros((N, 3), dtype=float)
    support_index_sequence = np.zeros(N, dtype=int)
    support_foot_sequence = []

    initial_support_foot = get_initial_support_foot(current_phase)
    support_index = 0
    previous_phase = current_phase

    for j, phase in enumerate(phase_sequence):
        if j > 0:
            support_index = advance_support_index_between_phases(
                previous_phase=previous_phase,
                current_phase=phase,
                support_index=support_index,
            )

        if support_index + 1 >= support_points_reanchored.shape[0]:
            raise ValueError(
                "Không đủ support_points để phủ toàn prediction horizon."
            )

        W_p_L, W_p_R = get_planned_feet_for_phase(
            phase=phase,
            support_index=support_index,
            support_points=support_points_reanchored,
            initial_support_foot=initial_support_foot,
        )

        W_p_L6_plan[j] = W_p_L
        W_p_R6_plan[j] = W_p_R
        support_index_sequence[j] = support_index
        support_foot_sequence.append(
            get_support_foot_from_index(
                initial_support_foot,
                support_index,
            )
        )

        previous_phase = phase

    # --------------------------------------------------------
    # Physical row-0 geometry for MPC.
    # --------------------------------------------------------
    if current_contact_meas is not None:
        current_contact_meas = _validate_contact2(
            current_contact_meas,
            "current_contact_meas",
        )

        if current_contact_meas[0] == 1:
            W_p_L6_plan[0] = W_p_L6_feedback

        if current_contact_meas[1] == 1:
            W_p_R6_plan[0] = W_p_R6_feedback

    return {
        "W_p_L6_plan": W_p_L6_plan,
        "W_p_R6_plan": W_p_R6_plan,
        "support_index_sequence": support_index_sequence,
        "support_foot_sequence": support_foot_sequence,
        "initial_support_foot": initial_support_foot,
        "support_points_reanchored": support_points_reanchored,
        "support_points_original": feedback_reanchor[
            "support_points_original"
        ],
        "delta_sequence": feedback_reanchor["delta_sequence"],
        "current_contact_meas": (
            None
            if current_contact_meas is None
            else current_contact_meas.copy()
        ),
    }


# ============================================================
# LTV MOMENT-ARM SEQUENCES
# ============================================================

def build_moment_arm_horizon(
    W_p_CoM_ref,
    W_p_L6_plan,
    W_p_R6_plan,
):
    """
    r_L,j = p_L6,j^plan - p_CoM,j^ref
    r_R,j = p_R6,j^plan - p_CoM,j^ref
    """
    W_p_CoM_ref = np.asarray(W_p_CoM_ref, dtype=float)
    W_p_L6_plan = np.asarray(W_p_L6_plan, dtype=float)
    W_p_R6_plan = np.asarray(W_p_R6_plan, dtype=float)

    if W_p_CoM_ref.ndim != 2 or W_p_CoM_ref.shape[1] != 3:
        raise ValueError("W_p_CoM_ref phải có shape (N,3).")
    if W_p_L6_plan.shape != W_p_CoM_ref.shape:
        raise ValueError(
            "W_p_L6_plan phải có cùng shape với W_p_CoM_ref."
        )
    if W_p_R6_plan.shape != W_p_CoM_ref.shape:
        raise ValueError(
            "W_p_R6_plan phải có cùng shape với W_p_CoM_ref."
        )

    if not (
        np.all(np.isfinite(W_p_CoM_ref))
        and np.all(np.isfinite(W_p_L6_plan))
        and np.all(np.isfinite(W_p_R6_plan))
    ):
        raise ValueError("Moment-arm inputs chứa NaN hoặc Inf.")

    W_r_L = W_p_L6_plan - W_p_CoM_ref
    W_r_R = W_p_R6_plan - W_p_CoM_ref

    return W_r_L, W_r_R
