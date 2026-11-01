import mujoco
import numpy as np


# ============================================================
# DEFAULT MUJOCO CONTACT OBJECT NAMES
#
# Current model:
#
#   ground geom:
#       floor
#
#   left foot collision body:
#       L5_ankle
#
#   right foot collision body:
#       R5_ankle
#
# IMPORTANT:
#
# L6 / R6 are SITES.
#
# MuJoCo contacts are GEOM-GEOM contacts, therefore contact
# detection must NOT try to use L6 / R6 site IDs.
# ============================================================

DEFAULT_FLOOR_GEOM_NAME = "floor"

DEFAULT_LEFT_FOOT_BODY_NAME = "L5_ankle"

DEFAULT_RIGHT_FOOT_BODY_NAME = "R5_ankle"


# ============================================================
# GET MUJOCO OBJECT ID
# ============================================================

def _get_mujoco_id(
    model,
    object_type,
    name,
    object_description
):
    object_id = mujoco.mj_name2id(
        model,
        object_type,
        name
    )


    if object_id < 0:

        raise ValueError(
            f"Không tìm thấy {object_description} "
            f"'{name}' trong MuJoCo model."
        )


    return int(
        object_id
    )


# ============================================================
# GET GEOM NAME
# ============================================================

def _get_geom_name(
    model,
    geom_id
):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        int(
            geom_id
        )
    )


    if name is None:

        return "<unnamed>"


    return str(
        name
    )


# ============================================================
# FIND CONTACT-CAPABLE GEOMS ATTACHED TO A BODY
# ============================================================

def _get_contact_geom_ids_from_body(
    model,
    body_id
):
    """
    Return all contact-capable geoms directly attached to body.

    A geom is considered contact-capable when at least one of:

        contype != 0
        conaffinity != 0

    This excludes the visual mesh geoms in the current model.
    """


    body_id = int(
        body_id
    )


    geom_ids = []


    for geom_id in range(
        model.ngeom
    ):

        if int(
            model.geom_bodyid[
                geom_id
            ]
        ) != body_id:

            continue


        contype = int(
            model.geom_contype[
                geom_id
            ]
        )


        conaffinity = int(
            model.geom_conaffinity[
                geom_id
            ]
        )


        if (
            contype != 0
            or
            conaffinity != 0
        ):

            geom_ids.append(
                int(
                    geom_id
                )
            )


    if len(
        geom_ids
    ) == 0:

        raise RuntimeError(
            "Không tìm thấy contact-capable geom "
            f"trên body ID {body_id}."
        )


    return tuple(
        geom_ids
    )


# ============================================================
# BUILD CONTACT STATE MAP
#
# Run this ONCE during simulation initialization.
#
# The resulting dictionary can then be reused at every
# simulation/control step.
# ============================================================

def build_contact_state_map(
    model,
    floor_geom_name=DEFAULT_FLOOR_GEOM_NAME,
    left_foot_body_name=DEFAULT_LEFT_FOOT_BODY_NAME,
    right_foot_body_name=DEFAULT_RIGHT_FOOT_BODY_NAME
):
    """
    Resolve MuJoCo IDs required for foot-ground contact
    detection.

    Returns
    -------
    contact_map : dict

        {
            "floor_geom_id": int,

            "left_foot_body_id": int,
            "right_foot_body_id": int,

            "left_foot_geom_ids": tuple[int],
            "right_foot_geom_ids": tuple[int],
        }
    """


    # ========================================================
    # 1. FLOOR GEOM
    # ========================================================

    floor_geom_id = _get_mujoco_id(

        model=model,

        object_type=(
            mujoco.mjtObj.mjOBJ_GEOM
        ),

        name=floor_geom_name,

        object_description="ground geom"
    )


    # ========================================================
    # 2. FOOT BODIES
    # ========================================================

    left_foot_body_id = _get_mujoco_id(

        model=model,

        object_type=(
            mujoco.mjtObj.mjOBJ_BODY
        ),

        name=left_foot_body_name,

        object_description="left foot body"
    )


    right_foot_body_id = _get_mujoco_id(

        model=model,

        object_type=(
            mujoco.mjtObj.mjOBJ_BODY
        ),

        name=right_foot_body_name,

        object_description="right foot body"
    )


    # ========================================================
    # 3. CONTACT-CAPABLE FOOT GEOMS
    # ========================================================

    left_foot_geom_ids = (
        _get_contact_geom_ids_from_body(

            model=model,

            body_id=left_foot_body_id
        )
    )


    right_foot_geom_ids = (
        _get_contact_geom_ids_from_body(

            model=model,

            body_id=right_foot_body_id
        )
    )


    # ========================================================
    # 4. SAFETY CHECK
    # ========================================================

    left_set = set(
        left_foot_geom_ids
    )


    right_set = set(
        right_foot_geom_ids
    )


    if not left_set.isdisjoint(
        right_set
    ):

        raise RuntimeError(
            "Left/right contact geom sets bị trùng."
        )


    # ========================================================
    # 5. RETURN
    # ========================================================

    return {

        "floor_geom_name":
            str(
                floor_geom_name
            ),

        "floor_geom_id":
            int(
                floor_geom_id
            ),

        "left_foot_body_name":
            str(
                left_foot_body_name
            ),

        "right_foot_body_name":
            str(
                right_foot_body_name
            ),

        "left_foot_body_id":
            int(
                left_foot_body_id
            ),

        "right_foot_body_id":
            int(
                right_foot_body_id
            ),

        "left_foot_geom_ids":
            tuple(
                left_foot_geom_ids
            ),

        "right_foot_geom_ids":
            tuple(
                right_foot_geom_ids
            ),
    }


# ============================================================
# CLASSIFY ONE MUJOCO CONTACT PAIR
#
# Return:
#
#   "L"
#   "R"
#   None
#
# Contact order does NOT matter:
#
#   floor -> foot
#
# and
#
#   foot -> floor
#
# are treated identically.
# ============================================================

def classify_contact_pair(
    geom1,
    geom2,
    contact_map
):
    geom1 = int(
        geom1
    )


    geom2 = int(
        geom2
    )


    floor_geom_id = int(
        contact_map[
            "floor_geom_id"
        ]
    )


    left_geom_ids = set(
        contact_map[
            "left_foot_geom_ids"
        ]
    )


    right_geom_ids = set(
        contact_map[
            "right_foot_geom_ids"
        ]
    )


    # ========================================================
    # CONTACT MUST INVOLVE FLOOR
    # ========================================================

    if geom1 == floor_geom_id:

        other_geom = (
            geom2
        )


    elif geom2 == floor_geom_id:

        other_geom = (
            geom1
        )


    else:

        return None


    # ========================================================
    # LEFT FOOT
    # ========================================================

    if other_geom in left_geom_ids:

        return "L"


    # ========================================================
    # RIGHT FOOT
    # ========================================================

    if other_geom in right_geom_ids:

        return "R"


    return None


# ============================================================
# BUILD CONTACT DIAGNOSTIC ENTRY
# ============================================================

def _build_contact_entry(
    model,
    contact_index,
    contact
):
    geom1 = int(
        contact.geom1
    )


    geom2 = int(
        contact.geom2
    )


    position = np.asarray(
        contact.pos,
        dtype=float
    ).copy()


    if position.shape != (
        3,
    ):

        raise RuntimeError(
            "MuJoCo contact position phải có shape (3,)."
        )


    return {

        "contact_index":
            int(
                contact_index
            ),

        "geom1_id":
            geom1,

        "geom2_id":
            geom2,

        "geom1_name":
            _get_geom_name(
                model,
                geom1
            ),

        "geom2_name":
            _get_geom_name(
                model,
                geom2
            ),

        "position":
            position,

        "distance":
            float(
                contact.dist
            ),
    }


# ============================================================
# GET CURRENT CONTACT STATE
#
# Main state-reading function.
#
# Input:
#
#   MuJoCo model/data at state x_k
#
# Output:
#
#   c_L(k)
#   c_R(k)
#
# where:
#
#   1 = foot is currently in contact with ground
#   0 = no foot-ground contact
# ============================================================

def get_contact_state(
    model,
    data,
    contact_map=None
):
    """
    Read current left/right physical ground contact state.

    Important
    ---------

    This function only READS the current MuJoCo contact state.

    It does NOT:

        - detect TD / LO edges
        - debounce contact
        - modify gait phase
        - advance simulation

    Those responsibilities belong to Command/contact_switch.py
    and Command/contact_schedule.py.


    Parameters
    ----------
    model : mujoco.MjModel

    data : mujoco.MjData

    contact_map : dict or None

        Result of:

            build_contact_state_map(model)

        For online simulation, build it once during
        initialization and reuse it.

        If None, the map is built automatically.


    Returns
    -------
    state : dict

        {
            "c_L": int,
            "c_R": int,

            "left_contact_count": int,
            "right_contact_count": int,

            "left_contacts": list,
            "right_contacts": list,

            "ncon": int
        }
    """


    # ========================================================
    # 1. CONTACT MAP
    # ========================================================

    if contact_map is None:

        contact_map = (
            build_contact_state_map(
                model
            )
        )


    # ========================================================
    # 2. CURRENT MUJOCO CONTACTS
    #
    # data.ncon belongs to the CURRENT MuJoCo state.
    #
    # Do NOT call mj_step() here.
    # ========================================================

    ncon = int(
        data.ncon
    )


    left_contacts = []

    right_contacts = []


    # ========================================================
    # 3. SCAN ACTIVE CONTACT PAIRS
    # ========================================================

    for contact_index in range(
        ncon
    ):

        contact = (
            data.contact[
                contact_index
            ]
        )


        side = classify_contact_pair(

            geom1=contact.geom1,

            geom2=contact.geom2,

            contact_map=contact_map
        )


        if side is None:

            continue


        entry = _build_contact_entry(

            model=model,

            contact_index=contact_index,

            contact=contact
        )


        if side == "L":

            left_contacts.append(
                entry
            )


        elif side == "R":

            right_contacts.append(
                entry
            )


        else:

            raise RuntimeError(
                "Unknown contact side."
            )


    # ========================================================
    # 4. BINARY CONTACT STATE
    # ========================================================

    c_L = int(
        len(
            left_contacts
        )
        >
        0
    )


    c_R = int(
        len(
            right_contacts
        )
        >
        0
    )


    # ========================================================
    # 5. OUTPUT
    # ========================================================

    return {

        "c_L":
            c_L,

        "c_R":
            c_R,

        "left_contact_count":
            int(
                len(
                    left_contacts
                )
            ),

        "right_contact_count":
            int(
                len(
                    right_contacts
                )
            ),

        "left_contacts":
            left_contacts,

        "right_contacts":
            right_contacts,

        "ncon":
            ncon,
    }


# ============================================================
# CONVENIENCE FUNCTION
#
# For the gait logic, this is often all we need:
#
#     c_L, c_R = get_binary_contact_state(...)
# ============================================================

def get_binary_contact_state(
    model,
    data,
    contact_map=None
):
    state = get_contact_state(

        model=model,

        data=data,

        contact_map=contact_map
    )


    return (
        int(
            state[
                "c_L"
            ]
        ),

        int(
            state[
                "c_R"
            ]
        )
    )