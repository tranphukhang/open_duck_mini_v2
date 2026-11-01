import numpy as np


# ============================================================
# NORMALIZE CONTACT STATE
# ============================================================

def normalize_contact_state(
    contact
):
    """
    Convert a contact value to binary form:

        0 = no contact

        1 = contact


    Accepted input
    --------------

    contact may be:

        bool

        int

        float


    Valid numerical values are:

        0

        1


    Parameters
    ----------
    contact : bool, int, float

        Contact state.


    Returns
    -------
    contact_binary : int

        Binary contact state:

            0 or 1
    """


    # ========================================================
    # BOOL
    # ========================================================

    if isinstance(
        contact,
        (bool, np.bool_)
    ):

        return int(
            contact
        )


    # ========================================================
    # NUMERICAL INPUT
    # ========================================================

    if isinstance(
        contact,
        (
            int,
            float,
            np.integer,
            np.floating
        )
    ):

        contact = float(
            contact
        )


        if not np.isfinite(
            contact
        ):

            raise ValueError(
                "Contact state chứa NaN hoặc Inf."
            )


        if np.isclose(
            contact,
            0.0
        ):

            return 0


        if np.isclose(
            contact,
            1.0
        ):

            return 1


        raise ValueError(
            "Contact state phải bằng 0 hoặc 1. "
            f"Nhận được {contact}."
        )


    # ========================================================
    # INVALID TYPE
    # ========================================================

    raise TypeError(
        "Contact state phải là bool hoặc giá trị số 0/1."
    )


# ============================================================
# DETECT ONE CONTACT EDGE
# ============================================================

def detect_contact_edge(
    previous_contact,
    current_contact
):
    """
    Detect touchdown and liftoff events for one foot.

    Edge definitions
    ----------------

    Touchdown:

        0 -> 1


    Liftoff:

        1 -> 0


    No event:

        0 -> 0

        1 -> 1


    Parameters
    ----------
    previous_contact : bool, int, float

        Contact state at the previous sample.


    current_contact : bool, int, float

        Contact state at the current sample.


    Returns
    -------
    touchdown : bool

        True only for:

            0 -> 1


    liftoff : bool

        True only for:

            1 -> 0
    """


    previous_contact = normalize_contact_state(
        previous_contact
    )


    current_contact = normalize_contact_state(
        current_contact
    )


    # ========================================================
    # TOUCHDOWN
    #
    #     0 -> 1
    # ========================================================

    touchdown = (
        previous_contact == 0
        and
        current_contact == 1
    )


    # ========================================================
    # LIFTOFF
    #
    #     1 -> 0
    # ========================================================

    liftoff = (
        previous_contact == 1
        and
        current_contact == 0
    )


    return (
        touchdown,
        liftoff
    )


# ============================================================
# DETECT LEFT / RIGHT CONTACT EVENTS
# ============================================================

def detect_contact_events(
    previous_left_contact,
    previous_right_contact,
    current_left_contact,
    current_right_contact
):
    """
    Detect contact-switch events for both feet.

    Contact convention
    ------------------

        c_L = 1
            Left foot is in contact.

        c_L = 0
            Left foot is not in contact.


        c_R = 1
            Right foot is in contact.

        c_R = 0
            Right foot is not in contact.


    Events
    ------

    Left touchdown:

        TD_L:
            c_L(k-1) = 0
            c_L(k)   = 1


    Right touchdown:

        TD_R:
            c_R(k-1) = 0
            c_R(k)   = 1


    Left liftoff:

        LO_L:
            c_L(k-1) = 1
            c_L(k)   = 0


    Right liftoff:

        LO_R:
            c_R(k-1) = 1
            c_R(k)   = 0


    Parameters
    ----------
    previous_left_contact : bool, int, float

        Previous left-foot contact state.


    previous_right_contact : bool, int, float

        Previous right-foot contact state.


    current_left_contact : bool, int, float

        Current left-foot contact state.


    current_right_contact : bool, int, float

        Current right-foot contact state.


    Returns
    -------
    events : dict

        Dictionary containing:

            events["TD_L"]

            events["TD_R"]

            events["LO_L"]

            events["LO_R"]

            events["c_L"]

            events["c_R"]


    Notes
    -----

    This function detects EDGES only.

    It does NOT trigger repeatedly while a foot remains
    in contact.

    Example:

        0 -> 1

        TD = True


    Next sample:

        1 -> 1

        TD = False


    This is important because gait-phase transitions must be
    triggered by contact EVENTS, not contact LEVELS.
    """


    # ========================================================
    # 1. NORMALIZE CONTACT STATES
    # ========================================================

    previous_left_contact = normalize_contact_state(
        previous_left_contact
    )


    previous_right_contact = normalize_contact_state(
        previous_right_contact
    )


    current_left_contact = normalize_contact_state(
        current_left_contact
    )


    current_right_contact = normalize_contact_state(
        current_right_contact
    )


    # ========================================================
    # 2. LEFT FOOT EVENTS
    # ========================================================

    (
        TD_L,
        LO_L

    ) = detect_contact_edge(

        previous_contact=previous_left_contact,

        current_contact=current_left_contact
    )


    # ========================================================
    # 3. RIGHT FOOT EVENTS
    # ========================================================

    (
        TD_R,
        LO_R

    ) = detect_contact_edge(

        previous_contact=previous_right_contact,

        current_contact=current_right_contact
    )


    # ========================================================
    # 4. OUTPUT
    # ========================================================

    events = {
        "TD_L": TD_L,

        "TD_R": TD_R,

        "LO_L": LO_L,

        "LO_R": LO_R,

        "c_L": current_left_contact,

        "c_R": current_right_contact,
    }


    return events


# ============================================================
# GET CURRENT PHYSICAL CONTACT MODE
# ============================================================

def get_contact_mode(
    left_contact,
    right_contact
):
    """
    Convert left/right contact states into a physical
    contact mode.

    Contact modes
    -------------

        [1, 0]
            -> "LSS"


        [0, 1]
            -> "RSS"


        [1, 1]
            -> "DS"


        [0, 0]
            -> "FLIGHT"


    Parameters
    ----------
    left_contact : bool, int, float

        Left-foot contact state.


    right_contact : bool, int, float

        Right-foot contact state.


    Returns
    -------
    mode : str

        One of:

            "LSS"

            "RSS"

            "DS"

            "FLIGHT"
    """


    left_contact = normalize_contact_state(
        left_contact
    )


    right_contact = normalize_contact_state(
        right_contact
    )


    # ========================================================
    # LEFT SINGLE SUPPORT
    # ========================================================

    if (
        left_contact == 1
        and
        right_contact == 0
    ):

        return "LSS"


    # ========================================================
    # RIGHT SINGLE SUPPORT
    # ========================================================

    if (
        left_contact == 0
        and
        right_contact == 1
    ):

        return "RSS"


    # ========================================================
    # DOUBLE SUPPORT
    # ========================================================

    if (
        left_contact == 1
        and
        right_contact == 1
    ):

        return "DS"


    # ========================================================
    # NO CONTACT
    # ========================================================

    return "FLIGHT"