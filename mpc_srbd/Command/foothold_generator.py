import numpy as np


# ============================================================
# KINEMATIC FOOTHOLD BOUNDS
#
# Source:
#   Simulation/run_kinematic_workspace_scan.py
#
# MuJoCo model:
#   scene_flat_terrian_v3.xml
#
# Reference configuration:
#   keyframe "home"
#
# IK:
#   numerical_IK_hybrid
#
# Workspace scan:
#
#   dx search:
#       [-0.15, +0.15] m
#
#   dy search:
#       [-0.10, +0.10] m
#
#   grid resolution:
#       0.005 m
#
# Safe joint-limit margin:
#       5 deg
#
# ------------------------------------------------------------
# GUARANTEED SAFE RECTANGLE
#
# Relative to the nominal / kinematic reference foot position.
#
# Longitudinal direction:
#
#   BOTH LEGS:
#
#       -0.055 <= dx <= +0.065 m
#
#
# Left foot lateral direction:
#
#       -0.025 <= dy_L <= +0.045 m
#
#
# Right foot lateral direction:
#
#       -0.045 <= dy_R <= +0.025 m
#
#
# Equivalent normalized lateral convention:
#
#       positive = outward
#
#       -0.025 <= dy_side <= +0.045 m
#
#
# IMPORTANT:
#
# These values are experimentally obtained KINEMATIC
# reachability bounds around the home-equivalent nominal foot
# position.
#
# They are NOT direct walking step-length limits.
#
# The nominal/reference foot position moves together with the
# robot body during walking.
# ============================================================


KINEMATIC_DX_MIN = -0.055
KINEMATIC_DX_MAX = +0.065


KINEMATIC_DY_BOUNDS = {

    "L": (
        -0.025,
        +0.045
    ),

    "R": (
        -0.045,
        +0.025
    ),
}


# ============================================================
# HOME NOMINAL STEP WIDTH
#
# Measured from MuJoCo home keyframe:
#
#   y_L6 = +0.054817 m
#
#   y_R6 = -0.054817 m
#
# Therefore:
#
#   W_home
#       =
#   y_L6 - y_R6
#       =
#   0.109634 m
#
# This is only a nominal/reference value.
#
# It is NOT a kinematic hard bound.
# ============================================================

HOME_NOMINAL_STEP_WIDTH = 0.109634


# ============================================================
# VALIDATE FOOT
# ============================================================

def normalize_foot(
    foot
):
    """
    Normalize foot label.

    Valid values:

        "L"
        "R"
    """


    foot = str(
        foot
    ).upper()


    if foot not in (
        "L",
        "R"
    ):

        raise ValueError(
            "foot phải là 'L' hoặc 'R'."
        )


    return foot


# ============================================================
# VALIDATE 3D VECTOR
# ============================================================

def _validate_vector3(
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


# ============================================================
# VALIDATE 2D VECTOR
# ============================================================

def _validate_vector2(
    value,
    name
):
    value = np.asarray(
        value,
        dtype=float
    ).reshape(-1)


    if value.shape != (
        2,
    ):

        raise ValueError(
            f"{name} phải có shape (2,), "
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


# ============================================================
# GET KINEMATIC BOUNDS
# ============================================================

def get_kinematic_foothold_bounds(
    swing_foot
):
    """
    Return the experimentally measured safe kinematic bounds.

    Returns
    -------
    bounds : dict

        {
            "dx_min": ...,
            "dx_max": ...,
            "dy_min": ...,
            "dy_max": ...
        }
    """


    swing_foot = normalize_foot(
        swing_foot
    )


    (
        dy_min,
        dy_max

    ) = KINEMATIC_DY_BOUNDS[
        swing_foot
    ]


    return {

        "dx_min":
            float(
                KINEMATIC_DX_MIN
            ),

        "dx_max":
            float(
                KINEMATIC_DX_MAX
            ),

        "dy_min":
            float(
                dy_min
            ),

        "dy_max":
            float(
                dy_max
            ),
    }


# ============================================================
# GET OPPOSITE FOOT
# ============================================================

def get_opposite_foot(
    foot
):
    foot = normalize_foot(
        foot
    )


    if foot == "L":

        return "R"


    return "L"


# ============================================================
# GENERATE NOMINAL FOOTHOLD
#
# Flat-ground walking.
#
# Nominal target is generated from the current support foot:
#
# Left swing:
#
#   x_L_nom = x_R_support + L_step
#
#   y_L_nom = y_R_support + W_step
#
#
# Right swing:
#
#   x_R_nom = x_L_support + L_step
#
#   y_R_nom = y_L_support - W_step
#
#
# Therefore:
#
#   step_length
#
# is the forward distance between consecutive footholds.
#
#
#   step_width
#
# is the lateral distance between left and right footholds.
# ============================================================

def generate_nominal_foothold(
    W_p_support,
    swing_foot,
    step_length,
    step_width=HOME_NOMINAL_STEP_WIDTH,
    ground_height=0.0
):
    """
    Generate a simple flat-ground nominal foothold.

    Parameters
    ----------
    W_p_support : array_like, shape (3,)

        Current support-foot position expressed in World.


    swing_foot : {"L", "R"}

        Foot that will move.


    step_length : float

        Longitudinal distance between consecutive footholds [m].

        Positive:
            forward walking.

        Negative:
            backward walking.


    step_width : float

        Lateral distance between left and right nominal
        footholds [m].


    ground_height : float

        Flat-ground z position [m].


    Returns
    -------
    W_p_nominal : np.ndarray, shape (3,)
    """


    W_p_support = _validate_vector3(
        W_p_support,
        "W_p_support"
    )


    swing_foot = normalize_foot(
        swing_foot
    )


    step_length = float(
        step_length
    )


    step_width = float(
        step_width
    )


    ground_height = float(
        ground_height
    )


    if not np.isfinite(
        step_length
    ):

        raise ValueError(
            "step_length phải hữu hạn."
        )


    if (
        not np.isfinite(
            step_width
        )
        or
        step_width <= 0.0
    ):

        raise ValueError(
            "step_width phải hữu hạn và > 0."
        )


    if not np.isfinite(
        ground_height
    ):

        raise ValueError(
            "ground_height phải hữu hạn."
        )


    W_p_nominal = np.zeros(
        3,
        dtype=float
    )


    # --------------------------------------------------------
    # Longitudinal nominal foothold
    # --------------------------------------------------------

    W_p_nominal[0] = (
        W_p_support[0]
        +
        step_length
    )


    # --------------------------------------------------------
    # Lateral nominal foothold
    # --------------------------------------------------------

    if swing_foot == "L":

        W_p_nominal[1] = (
            W_p_support[1]
            +
            step_width
        )


    else:

        W_p_nominal[1] = (
            W_p_support[1]
            -
            step_width
        )


    # --------------------------------------------------------
    # Flat ground
    # --------------------------------------------------------

    W_p_nominal[2] = (
        ground_height
    )


    return W_p_nominal


# ============================================================
# APPLY PLANNER CORRECTION
# ============================================================

def apply_foothold_correction(
    W_p_nominal,
    correction_xy
):
    """
    Add a horizontal planner correction to nominal foothold.

        p_candidate
            =
        p_nominal
            +
        [dx_corr, dy_corr, 0]

    This function does NOT perform kinematic clipping.
    """


    W_p_nominal = _validate_vector3(
        W_p_nominal,
        "W_p_nominal"
    )


    correction_xy = _validate_vector2(
        correction_xy,
        "correction_xy"
    )


    W_p_candidate = (
        W_p_nominal.copy()
    )


    W_p_candidate[0] += (
        correction_xy[0]
    )


    W_p_candidate[1] += (
        correction_xy[1]
    )


    return W_p_candidate


# ============================================================
# CLIP FOOTHOLD TO MEASURED KINEMATIC WORKSPACE
#
# The final foothold is bounded relative to:
#
#     W_p_kinematic_reference
#
# where this reference should represent the current
# home-equivalent / nominal foot location for the current
# robot-body pose.
#
#
# Delta:
#
#     delta_p
#         =
#     p_candidate
#         -
#     p_kinematic_reference
#
#
# The measured bounds are then applied to delta_x and delta_y.
# ============================================================

def clip_foothold_to_kinematic_workspace(
    W_p_candidate,
    W_p_kinematic_reference,
    swing_foot,
    ground_height=0.0
):
    """
    Clip candidate foothold to experimentally measured
    safe kinematic workspace.

    Returns
    -------
    W_p_plan : np.ndarray, shape (3,)

        Bounded foothold.


    diagnostics : dict

        Information about requested and bounded displacement.
    """


    W_p_candidate = _validate_vector3(
        W_p_candidate,
        "W_p_candidate"
    )


    W_p_kinematic_reference = _validate_vector3(
        W_p_kinematic_reference,
        "W_p_kinematic_reference"
    )


    swing_foot = normalize_foot(
        swing_foot
    )


    ground_height = float(
        ground_height
    )


    if not np.isfinite(
        ground_height
    ):

        raise ValueError(
            "ground_height phải hữu hạn."
        )


    bounds = (
        get_kinematic_foothold_bounds(
            swing_foot
        )
    )


    # ========================================================
    # REQUESTED DISPLACEMENT FROM KINEMATIC REFERENCE
    # ========================================================

    dx_requested = float(
        W_p_candidate[0]
        -
        W_p_kinematic_reference[0]
    )


    dy_requested = float(
        W_p_candidate[1]
        -
        W_p_kinematic_reference[1]
    )


    # ========================================================
    # CLIP TO MEASURED SAFE RECTANGLE
    # ========================================================

    dx_bounded = float(
        np.clip(
            dx_requested,
            bounds[
                "dx_min"
            ],
            bounds[
                "dx_max"
            ]
        )
    )


    dy_bounded = float(
        np.clip(
            dy_requested,
            bounds[
                "dy_min"
            ],
            bounds[
                "dy_max"
            ]
        )
    )


    # ========================================================
    # FINAL PLANNED FOOTHOLD
    # ========================================================

    W_p_plan = np.array(
        [
            W_p_kinematic_reference[0]
            +
            dx_bounded,

            W_p_kinematic_reference[1]
            +
            dy_bounded,

            ground_height
        ],
        dtype=float
    )


    clipped_x = bool(
        not np.isclose(
            dx_requested,
            dx_bounded,
            atol=1e-12,
            rtol=1e-12
        )
    )


    clipped_y = bool(
        not np.isclose(
            dy_requested,
            dy_bounded,
            atol=1e-12,
            rtol=1e-12
        )
    )


    diagnostics = {

        "swing_foot":
            swing_foot,

        "bounds":
            bounds,

        "dx_requested":
            dx_requested,

        "dy_requested":
            dy_requested,

        "dx_bounded":
            dx_bounded,

        "dy_bounded":
            dy_bounded,

        "clipped_x":
            clipped_x,

        "clipped_y":
            clipped_y,

        "clipped":
            bool(
                clipped_x
                or
                clipped_y
            ),
    }


    return (
        W_p_plan,
        diagnostics
    )


# ============================================================
# COMPLETE FOOTHOLD GENERATOR
# ============================================================

def generate_foothold(
    W_p_support,
    W_p_kinematic_reference,
    swing_foot,
    step_length,
    step_width=HOME_NOMINAL_STEP_WIDTH,
    correction_xy=None,
    ground_height=0.0
):
    """
    Complete flat-ground foothold generator.

    Pipeline
    --------

        support foot
            ->
        nominal foothold
            ->
        planner correction
            ->
        kinematic workspace clipping
            ->
        planned foothold


    Parameters
    ----------
    W_p_support : shape (3,)

        Current support-foot position in World.


    W_p_kinematic_reference : shape (3,)

        Current nominal/home-equivalent location of the
        SWING foot.

        IMPORTANT:

        The safe workspace measured from the home keyframe is
        relative to this reference.

        Therefore this reference should move with the robot
        body during walking.


    swing_foot : {"L", "R"}

        Swing leg.


    step_length : float

        Nominal longitudinal distance between consecutive
        footholds [m].


    step_width : float

        Nominal lateral distance between left and right
        footholds [m].


    correction_xy : array_like, shape (2,), optional

        Additional foothold correction generated by a higher
        level planner.

        Examples later:

            DCM correction

            Raibert correction

            RL correction

        If None:

            [0, 0]


    ground_height : float

        Flat-ground height [m].


    Returns
    -------
    W_p_plan : np.ndarray, shape (3,)

        Final kinematically bounded foothold.


    diagnostics : dict
    """


    if correction_xy is None:

        correction_xy = np.zeros(
            2,
            dtype=float
        )


    # ========================================================
    # 1. NOMINAL FOOTHOLD
    # ========================================================

    W_p_nominal = (
        generate_nominal_foothold(

            W_p_support=W_p_support,

            swing_foot=swing_foot,

            step_length=step_length,

            step_width=step_width,

            ground_height=ground_height
        )
    )


    # ========================================================
    # 2. HIGHER-LEVEL PLANNER CORRECTION
    # ========================================================

    W_p_candidate = (
        apply_foothold_correction(

            W_p_nominal=W_p_nominal,

            correction_xy=correction_xy
        )
    )


    # ========================================================
    # 3. KINEMATIC SAFETY CLIPPING
    # ========================================================

    (
        W_p_plan,
        kinematic_diagnostics

    ) = clip_foothold_to_kinematic_workspace(

        W_p_candidate=W_p_candidate,

        W_p_kinematic_reference=(
            W_p_kinematic_reference
        ),

        swing_foot=swing_foot,

        ground_height=ground_height
    )


    # ========================================================
    # 4. COMPLETE DIAGNOSTICS
    # ========================================================

    diagnostics = {

        "swing_foot":
            normalize_foot(
                swing_foot
            ),

        "W_p_nominal":
            W_p_nominal.copy(),

        "correction_xy":
            _validate_vector2(
                correction_xy,
                "correction_xy"
            ).copy(),

        "W_p_candidate":
            W_p_candidate.copy(),

        "W_p_kinematic_reference":
            _validate_vector3(
                W_p_kinematic_reference,
                "W_p_kinematic_reference"
            ).copy(),

        "kinematic":
            kinematic_diagnostics,
    }


    return (
        W_p_plan,
        diagnostics
    )