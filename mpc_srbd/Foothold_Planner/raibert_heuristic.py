import numpy as np


# ============================================================
# RAIBERT HEURISTIC FOOTHOLD PLANNER
# ============================================================

def compute_raibert_foothold(
    W_p_L6_ref,
    W_p_R6_ref,
    W_p_dot_CoM,
    W_p_dot_CoM_ref,
    swing_foot,
    delta_T,
    k_d=0.05,
    ground_height=0.0
):
    """
    Compute the desired foothold position of the current
    swing foot using the Raibert heuristic.

    The foothold is expressed in the World frame.

    For the current implementation:

        x and y:

            computed using the Raibert heuristic

        z:

            assigned directly from ground_height


    Raibert heuristic
    -----------------

    For swing foot i:

        {}^W p_foothold,i
            =
        {}^W p_i6,ref
            +
        1/2 * {}^W p_dot_CoM * delta_T
            +
        k_d
        (
            {}^W p_dot_CoM
            -
            {}^W p_dot_CoM^ref
        )


    In the current flat-terrain implementation:

        {}^W z_foothold = ground_height


    Parameters
    ----------
    W_p_L6_ref : array_like, shape (3,)

        Nominal left-foot reference position expressed
        in the World frame:

            {}^W p_L6,ref


    W_p_R6_ref : array_like, shape (3,)

        Nominal right-foot reference position expressed
        in the World frame:

            {}^W p_R6,ref


    W_p_dot_CoM : array_like, shape (3,)

        Current whole-body CoM linear velocity expressed
        in the World frame:

            {}^W p_dot_CoM

        Unit:

            [m/s]


    W_p_dot_CoM_ref : array_like, shape (3,)

        Reference whole-body CoM linear velocity expressed
        in the World frame:

            {}^W p_dot_CoM^ref

        Unit:

            [m/s]


    swing_foot : str or None

        Current swing-foot identity:

            "L"  -> left foot is swing

            "R"  -> right foot is swing

            None -> Double Support


    delta_T : float or None

        Remaining time until the current swing phase ends.

        Unit:

            [s]

        During Double Support:

            delta_T = None


    k_d : float, optional

        Raibert velocity-feedback gain.

        Default:

            0.05


    ground_height : float, optional

        Desired touchdown ground height in the World frame.

        Default:

            0.0 m


    Returns
    -------
    W_p_foothold : np.ndarray, shape (3,), or None

        Desired foothold position of the current swing foot:

            [
                x_foothold,
                y_foothold,
                z_foothold
            ]

        expressed in the World frame.

        Returns None during Double Support.
    """


    # ========================================================
    # 1. CONVERT INPUTS
    # ========================================================

    W_p_L6_ref = np.asarray(
        W_p_L6_ref,
        dtype=float
    ).reshape(-1)


    W_p_R6_ref = np.asarray(
        W_p_R6_ref,
        dtype=float
    ).reshape(-1)


    W_p_dot_CoM = np.asarray(
        W_p_dot_CoM,
        dtype=float
    ).reshape(-1)


    W_p_dot_CoM_ref = np.asarray(
        W_p_dot_CoM_ref,
        dtype=float
    ).reshape(-1)


    # ========================================================
    # 2. INPUT VALIDATION
    # ========================================================

    for name, value in (

        (
            "W_p_L6_ref",
            W_p_L6_ref
        ),

        (
            "W_p_R6_ref",
            W_p_R6_ref
        ),

        (
            "W_p_dot_CoM",
            W_p_dot_CoM
        ),

        (
            "W_p_dot_CoM_ref",
            W_p_dot_CoM_ref
        )

    ):

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


    # --------------------------------------------------------
    # Validate k_d
    # --------------------------------------------------------

    k_d = float(
        k_d
    )


    if not np.isfinite(
        k_d
    ):

        raise ValueError(
            "k_d phải finite."
        )


    if k_d < 0.0:

        raise ValueError(
            "k_d không được nhỏ hơn 0."
        )


    # --------------------------------------------------------
    # Validate ground height
    # --------------------------------------------------------

    ground_height = float(
        ground_height
    )


    if not np.isfinite(
        ground_height
    ):

        raise ValueError(
            "ground_height phải finite."
        )


    # ========================================================
    # 3. DOUBLE SUPPORT
    #
    # During Double Support:
    #
    #     swing_foot = None
    #
    # There is no swing-foot foothold to compute.
    # ========================================================

    if swing_foot is None:

        if delta_T is not None:

            raise ValueError(
                "delta_T phải là None khi swing_foot là None."
            )

        return None


    # ========================================================
    # 4. VALIDATE SWING FOOT
    # ========================================================

    if swing_foot not in (
        "L",
        "R"
    ):

        raise ValueError(
            "swing_foot chỉ được là 'L', 'R' hoặc None."
        )


    # ========================================================
    # 5. VALIDATE REMAINING SWING TIME
    # ========================================================

    if delta_T is None:

        raise ValueError(
            "delta_T không được là None khi đang có swing foot."
        )


    delta_T = float(
        delta_T
    )


    if not np.isfinite(
        delta_T
    ):

        raise ValueError(
            "delta_T phải finite."
        )


    if delta_T <= 0.0:

        raise ValueError(
            "delta_T phải lớn hơn 0 khi đang ở swing phase."
        )


    # ========================================================
    # 6. SELECT NOMINAL FOOT REFERENCE
    # ========================================================

    if swing_foot == "L":

        W_p_foot_ref = (
            W_p_L6_ref.copy()
        )


    else:

        W_p_foot_ref = (
            W_p_R6_ref.copy()
        )


    # ========================================================
    # 7. RAIBERT HEURISTIC
    #
    # Only x-y components are used for foothold planning.
    #
    # Feedforward term:
    #
    #     1/2 * W_p_dot_CoM * delta_T
    #
    # Velocity-feedback term:
    #
    #     k_d
    #     (
    #         W_p_dot_CoM
    #         -
    #         W_p_dot_CoM_ref
    #     )
    # ========================================================

    W_p_foothold = (
        W_p_foot_ref.copy()
    )


    # --------------------------------------------------------
    # x-y foothold
    # --------------------------------------------------------

    W_p_foothold[
        0:2
    ] = (

        W_p_foot_ref[
            0:2
        ]

        +

        0.5
        *
        W_p_dot_CoM[
            0:2
        ]
        *
        delta_T

        +

        k_d
        *
        (
            W_p_dot_CoM[
                0:2
            ]

            -

            W_p_dot_CoM_ref[
                0:2
            ]
        )

    )


    # --------------------------------------------------------
    # z foothold
    #
    # Flat terrain:
    #
    #     z_foothold = ground_height
    # --------------------------------------------------------

    W_p_foothold[
        2
    ] = ground_height


    # ========================================================
    # 8. OUTPUT VALIDATION
    # ========================================================

    if W_p_foothold.shape != (
        3,
    ):

        raise RuntimeError(
            "W_p_foothold phải có shape (3,), "
            f"nhưng nhận được {W_p_foothold.shape}."
        )


    if not np.all(
        np.isfinite(
            W_p_foothold
        )
    ):

        raise RuntimeError(
            "W_p_foothold chứa NaN hoặc Inf."
        )


    # ========================================================
    # 9. RETURN
    # ========================================================

    return W_p_foothold