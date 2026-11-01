import numpy as np

from Command.contact_schedule import (
    DOUBLE_SUPPORT
)


# ============================================================
# INITIAL STANDING
# ============================================================

def get_initial_standing_info(
    current_time,
    T_stand,
    N
):
    """
    Get the startup standing-mode information.

    During:

        current_time < T_stand

    the robot is forced to remain in Double Support:

        current_contact = [1, 1]

    No swing leg exists:

        swing_foot = None

        delta_T = None

    The complete MPC prediction horizon is also forced to
    Double Support:

        contact_schedule =
        [
            [1, 1],
            [1, 1],
            ...
            [1, 1]
        ]

    This helper is intended to be reused by different
    simulation run files before their main locomotion task
    begins.


    Parameters
    ----------
    current_time : float

        Current simulation time [s].


    T_stand : float

        Initial standing duration [s].


    N : int

        MPC prediction horizon.


    Returns
    -------
    is_standing : bool

        True:

            current_time < T_stand

        False:

            current_time >= T_stand


    current_contact : np.ndarray or None

        During initial standing:

            [1, 1]

        Otherwise:

            None


    swing_foot : None

        During initial standing there is no swing foot.


    delta_T : None

        No swing phase exists during initial standing.


    contact_schedule : np.ndarray or None

        Shape:

            (N, 2)

        During initial standing the complete horizon is:

            Double Support

        Otherwise:

            None
    """


    # ========================================================
    # 1. INPUT VALIDATION
    # ========================================================

    current_time = float(
        current_time
    )


    T_stand = float(
        T_stand
    )


    if not np.isfinite(
        current_time
    ):

        raise ValueError(
            "current_time phải finite."
        )


    if current_time < 0.0:

        raise ValueError(
            "current_time không được nhỏ hơn 0."
        )


    if not np.isfinite(
        T_stand
    ):

        raise ValueError(
            "T_stand phải finite."
        )


    if T_stand < 0.0:

        raise ValueError(
            "T_stand phải >= 0."
        )


    if not isinstance(
        N,
        (int, np.integer)
    ):

        raise TypeError(
            "N phải là số nguyên."
        )


    if N <= 0:

        raise ValueError(
            "N phải lớn hơn 0."
        )


    # ========================================================
    # 2. CHECK INITIAL STANDING PHASE
    # ========================================================

    is_standing = (
        current_time
        <
        T_stand
    )


    # ========================================================
    # 3. INITIAL STANDING ACTIVE
    # ========================================================

    if is_standing:

        # ----------------------------------------------------
        # Current contact:
        #
        #     [1, 1]
        #
        # Both feet are stance.
        # ----------------------------------------------------

        current_contact = (
            DOUBLE_SUPPORT.copy()
        )


        # ----------------------------------------------------
        # No swing leg
        # ----------------------------------------------------

        swing_foot = None


        # ----------------------------------------------------
        # No remaining swing time
        # ----------------------------------------------------

        delta_T = None


        # ----------------------------------------------------
        # Force the entire MPC horizon to Double Support.
        #
        # Shape:
        #
        #     (N, 2)
        # ----------------------------------------------------

        contact_schedule = np.tile(
            DOUBLE_SUPPORT,
            (
                N,
                1
            )
        )


    # ========================================================
    # 4. INITIAL STANDING FINISHED
    #
    # The run file should switch to its task-specific contact
    # logic, for example:
    #
    #     walking scheduler
    #     turning scheduler
    #     another locomotion mode
    # ========================================================

    else:

        current_contact = None

        swing_foot = None

        delta_T = None

        contact_schedule = None


    # ========================================================
    # 5. RETURN
    # ========================================================

    return (
        is_standing,
        current_contact,
        swing_foot,
        delta_T,
        contact_schedule
    )