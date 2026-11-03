import numpy as np


# ============================================================
# SRBD STATE DEFINITION
#
# x =
#
# [
#   pCoM_W,       3
#   eulerCoM_W,   3
#   vCoM_W,       3
#   wCoM_W,       3
#   1             1
# ]
#
# Total:
#
#   nx = 13
#
# Index:
#
#   0 : x
#   1 : y
#   2 : z
#
#   3 : roll
#   4 : pitch
#   5 : yaw
#
#   6 : vx
#   7 : vy
#   8 : vz
#
#   9  : wx
#   10 : wy
#   11 : wz
#
#   12 : homogeneous constant = 1
# ============================================================

NX = 13


# ============================================================
# STATE INDICES
# ============================================================

IDX_POSITION = slice(
    0,
    3,
)

IDX_EULER = slice(
    3,
    6,
)

IDX_LINEAR_VELOCITY = slice(
    6,
    9,
)

IDX_ANGULAR_VELOCITY = slice(
    9,
    12,
)

IDX_CONSTANT = 12


# ============================================================
# DEFAULT PARAMETERS
#
# Paper-style values:
#
# position reference guard:
#     0.1 m
#
# yaw reference guard:
#     0.3 rad
#
# velocity threshold:
#     0.01 m/s
#
# yaw-rate threshold:
#     0.01 rad/s
#
# yaw singularity margin:
#     pi / 36 = 5 deg
# ============================================================

DEFAULT_MAX_POSITION_ERROR = 0.10

DEFAULT_MAX_YAW_ERROR = 0.30

DEFAULT_VELOCITY_THRESHOLD = 0.01

DEFAULT_YAW_RATE_THRESHOLD = 0.01

DEFAULT_YAW_RESET_MARGIN = (
    np.pi
    /
    36.0
)


# ============================================================
# VECTOR VALIDATION
# ============================================================

def _validate_vector(
    value,
    name,
    shape,
):

    value = np.asarray(
        value,
        dtype=float,
    ).reshape(-1)

    if value.shape != shape:

        raise ValueError(
            f"{name} must have shape {shape}, "
            f"but received {value.shape}."
        )

    if not np.all(
        np.isfinite(
            value
        )
    ):

        raise ValueError(
            f"{name} contains NaN or Inf."
        )

    return value.copy()


# ============================================================
# SCALAR VALIDATION
# ============================================================

def _validate_positive_scalar(
    value,
    name,
):

    value = float(
        value
    )

    if (
        not np.isfinite(
            value
        )
        or
        value <= 0.0
    ):

        raise ValueError(
            f"{name} must be finite and > 0."
        )

    return value


def _validate_nonnegative_scalar(
    value,
    name,
):

    value = float(
        value
    )

    if (
        not np.isfinite(
            value
        )
        or
        value < 0.0
    ):

        raise ValueError(
            f"{name} must be finite and >= 0."
        )

    return value


# ============================================================
# ROTATION MATRIX
#
# Euler convention:
#
#     roll  = phi
#     pitch = theta
#     yaw   = psi
#
# World orientation:
#
#     R_WB
#
# maps a vector from Body frame to World frame:
#
#     v_W = R_WB @ v_B
#
# Standard ZYX:
#
#     R_WB = Rz(yaw) Ry(pitch) Rx(roll)
# ============================================================

def euler_to_rotation_matrix(
    euler,
):

    euler = _validate_vector(
        euler,
        "euler",
        (3,),
    )

    roll = float(
        euler[0]
    )

    pitch = float(
        euler[1]
    )

    yaw = float(
        euler[2]
    )

    cr = np.cos(
        roll
    )

    sr = np.sin(
        roll
    )

    cp = np.cos(
        pitch
    )

    sp = np.sin(
        pitch
    )

    cy = np.cos(
        yaw
    )

    sy = np.sin(
        yaw
    )


    R_WB = np.array(
        [
            [
                cy * cp,
                cy * sp * sr - sy * cr,
                cy * sp * cr + sy * sr,
            ],

            [
                sy * cp,
                sy * sp * sr + cy * cr,
                sy * sp * cr - cy * sr,
            ],

            [
                -sp,
                cp * sr,
                cp * cr,
            ],
        ],
        dtype=float,
    )

    return R_WB


# ============================================================
# REFERENCE GENERATOR
# ============================================================

class ReferenceGenerator:
    """
    Paper-style SRBD reference trajectory generator.

    Inputs
    ------

    x_actual : shape (13,)

        Current SRBD state:

        [
            pCoM_W,
            eulerCoM_W,
            vCoM_W,
            wCoM_W,
            1
        ]


    v_des_body : shape (3,)

        Desired body-frame translational velocity:

        [
            vx_des_B,
            vy_des_B,
            vz_des_B
        ]


    yaw_rate_des : float

        Desired yaw rate [rad/s].


    z_des : float

        Desired CoM/body height [m].


    roll_des : float

        Desired roll angle [rad].


    pitch_des : float

        Desired pitch angle [rad].


    Output
    ------

    X_ref : shape (N, 13)

        SRBD reference trajectory over the MPC horizon.


    Important
    ---------

    reference_update_dt:

        Time interval between two calls that UPDATE the
        persistent position/yaw command.

        This corresponds to:

            mpc_decimation * dt

        in the paper.


    dt_mpc:

        Prediction sampling time inside the MPC horizon.

        This is NOT necessarily equal to reference_update_dt.
    """

    def __init__(
        self,
        horizon_length=10,
        dt_mpc=0.04,
        reference_update_dt=0.005,
        max_position_error=
            DEFAULT_MAX_POSITION_ERROR,
        max_yaw_error=
            DEFAULT_MAX_YAW_ERROR,
        velocity_threshold=
            DEFAULT_VELOCITY_THRESHOLD,
        yaw_rate_threshold=
            DEFAULT_YAW_RATE_THRESHOLD,
        yaw_reset_margin=
            DEFAULT_YAW_RESET_MARGIN,
    ):

        # ====================================================
        # BASIC PARAMETERS
        # ====================================================

        self.horizon_length = int(
            horizon_length
        )

        if self.horizon_length <= 0:

            raise ValueError(
                "horizon_length must be > 0."
            )


        self.dt_mpc = (
            _validate_positive_scalar(
                dt_mpc,
                "dt_mpc",
            )
        )


        self.reference_update_dt = (
            _validate_positive_scalar(
                reference_update_dt,
                "reference_update_dt",
            )
        )


        self.max_position_error = (
            _validate_nonnegative_scalar(
                max_position_error,
                "max_position_error",
            )
        )


        self.max_yaw_error = (
            _validate_nonnegative_scalar(
                max_yaw_error,
                "max_yaw_error",
            )
        )


        self.velocity_threshold = (
            _validate_nonnegative_scalar(
                velocity_threshold,
                "velocity_threshold",
            )
        )


        self.yaw_rate_threshold = (
            _validate_nonnegative_scalar(
                yaw_rate_threshold,
                "yaw_rate_threshold",
            )
        )


        self.yaw_reset_margin = (
            _validate_nonnegative_scalar(
                yaw_reset_margin,
                "yaw_reset_margin",
            )
        )


        # ====================================================
        # INTERNAL STATE
        # ====================================================

        self.first_run = True


        self.world_position_desired = np.zeros(
            3,
            dtype=float,
        )


        self.yaw_desired = 0.0


        # ====================================================
        # LAST OUTPUT
        # ====================================================

        self.last_output = None


    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        x_actual=None,
    ):

        self.first_run = True

        self.world_position_desired[:] = 0.0

        self.yaw_desired = 0.0

        self.last_output = None


        # ----------------------------------------------------
        # Optional immediate initialization.
        # ----------------------------------------------------

        if x_actual is not None:

            x_actual = self._validate_state(
                x_actual
            )

            self._initialize_from_state(
                x_actual
            )


    # ========================================================
    # STATE VALIDATION
    # ========================================================

    def _validate_state(
        self,
        x_actual,
    ):

        return _validate_vector(
            x_actual,
            "x_actual",
            (NX,),
        )


    # ========================================================
    # INITIALIZE FROM CURRENT ROBOT STATE
    #
    # Paper:
    #
    # world_position_desired[0] = actual x
    # world_position_desired[1] = actual y
    # yaw_desired               = actual yaw
    # ========================================================

    def _initialize_from_state(
        self,
        x_actual,
    ):

        self.world_position_desired[
            0
        ] = x_actual[
            0
        ]

        self.world_position_desired[
            1
        ] = x_actual[
            1
        ]

        self.world_position_desired[
            2
        ] = x_actual[
            2
        ]

        self.yaw_desired = float(
            x_actual[
                5
            ]
        )

        self.first_run = False


    # ========================================================
    # BODY -> WORLD VELOCITY
    # ========================================================

    def body_velocity_to_world(
        self,
        x_actual,
        v_des_body,
    ):

        x_actual = self._validate_state(
            x_actual
        )

        v_des_body = _validate_vector(
            v_des_body,
            "v_des_body",
            (3,),
        )


        euler_actual = x_actual[
            IDX_EULER
        ]


        R_WB = (
            euler_to_rotation_matrix(
                euler_actual
            )
        )


        v_des_world = (

            R_WB

            @

            v_des_body
        )


        return v_des_world


    # ========================================================
    # YAW KNOT PROTECTION
    #
    # Same idea as paper:
    #
    # if desired yaw gets close to +/- pi,
    # reset the knot to the measured yaw.
    # ========================================================

    def _apply_yaw_boundary_protection(
        self,
        yaw_actual,
    ):

        upper_limit = (

            np.pi

            -

            self.yaw_reset_margin
        )

        lower_limit = (

            -np.pi

            +

            self.yaw_reset_margin
        )


        if (
            self.yaw_desired
            >
            upper_limit
            or
            self.yaw_desired
            <
            lower_limit
        ):

            self.yaw_desired = float(
                yaw_actual
            )


    # ========================================================
    # UPDATE PERSISTENT COMMAND
    #
    # Paper:
    #
    # x_des += update_dt * vx_des_world
    #
    # y_des += update_dt * vy_des_world
    #
    # z_des  = commanded height
    #
    # yaw_des += update_dt * yaw_rate_des
    # ========================================================

    def _update_persistent_command(
        self,
        v_des_world,
        yaw_rate_des,
        z_des,
    ):

        self.world_position_desired[
            0
        ] += (

            self.reference_update_dt

            *

            v_des_world[
                0
            ]
        )


        self.world_position_desired[
            1
        ] += (

            self.reference_update_dt

            *

            v_des_world[
                1
            ]
        )


        # Paper uses direct height command.
        self.world_position_desired[
            2
        ] = float(
            z_des
        )


        self.yaw_desired += (

            self.reference_update_dt

            *

            float(
                yaw_rate_des
            )
        )


    # ========================================================
    # ERROR GUARD
    #
    # Important:
    #
    # This does NOT overwrite the persistent desired position.
    #
    # It only chooses a safe starting knot for the current
    # MPC reference horizon, following the paper.
    # ========================================================

    def _compute_safe_start(
        self,
        x_actual,
    ):

        x_actual_pos = float(
            x_actual[
                0
            ]
        )

        y_actual_pos = float(
            x_actual[
                1
            ]
        )

        yaw_actual = float(
            x_actual[
                5
            ]
        )


        x_start = float(
            self.world_position_desired[
                0
            ]
        )

        y_start = float(
            self.world_position_desired[
                1
            ]
        )

        yaw_start = float(
            self.yaw_desired
        )


        # ====================================================
        # X GUARD
        # ====================================================

        if (
            abs(
                x_start
                -
                x_actual_pos
            )
            >
            self.max_position_error
        ):

            x_start = (
                x_actual_pos
            )


        # ====================================================
        # Y GUARD
        # ====================================================

        if (
            abs(
                y_start
                -
                y_actual_pos
            )
            >
            self.max_position_error
        ):

            y_start = (
                y_actual_pos
            )


        # ====================================================
        # YAW GUARD
        # ====================================================

        if (
            abs(
                yaw_start
                -
                yaw_actual
            )
            >
            self.max_yaw_error
        ):

            yaw_start = (
                yaw_actual
            )


        return (
            float(
                x_start
            ),

            float(
                y_start
            ),

            float(
                yaw_start
            ),
        )


    # ========================================================
    # BUILD ONE INITIAL REFERENCE STATE
    #
    # Paper state:
    #
    # [roll pitch yaw x y z wx wy wz vx vy vz]
    #
    # Our state:
    #
    # [x y z roll pitch yaw vx vy vz wx wy wz 1]
    # ========================================================

    def _build_initial_reference(
        self,
        x_start,
        y_start,
        yaw_start,
        z_des,
        roll_des,
        pitch_des,
        v_des_world,
        yaw_rate_des,
    ):

        x_ref = np.zeros(
            NX,
            dtype=float,
        )


        # ====================================================
        # POSITION
        # ====================================================

        x_ref[
            0
        ] = x_start

        x_ref[
            1
        ] = y_start

        x_ref[
            2
        ] = z_des


        # ====================================================
        # EULER
        # ====================================================

        x_ref[
            3
        ] = roll_des

        x_ref[
            4
        ] = pitch_des

        x_ref[
            5
        ] = yaw_start


        # ====================================================
        # LINEAR VELOCITY
        # ====================================================

        x_ref[
            6:9
        ] = v_des_world


        # ====================================================
        # ANGULAR VELOCITY
        #
        # roll rate  = 0
        # pitch rate = 0
        # yaw rate   = commanded yaw rate
        # ====================================================

        x_ref[
            9
        ] = 0.0

        x_ref[
            10
        ] = 0.0

        x_ref[
            11
        ] = yaw_rate_des


        # ====================================================
        # HOMOGENEOUS CONSTANT
        # ====================================================

        x_ref[
            IDX_CONSTANT
        ] = 1.0


        return x_ref


    # ========================================================
    # BUILD MPC HORIZON
    # ========================================================

    def _build_horizon(
        self,
        x_actual,
        initial_reference,
        v_des_world,
        yaw_rate_des,
        x_start,
        y_start,
        yaw_start,
    ):

        X_ref = np.tile(
            initial_reference,
            (
                self.horizon_length,
                1,
            ),
        )


        x_actual_pos = float(
            x_actual[
                0
            ]
        )

        y_actual_pos = float(
            x_actual[
                1
            ]
        )

        yaw_actual = float(
            x_actual[
                5
            ]
        )


        for i in range(
            self.horizon_length
        ):

            prediction_time = (

                float(
                    i
                )

                *

                self.dt_mpc
            )


            # =================================================
            # DEFAULT MOVING REFERENCE
            #
            # Anchor horizon to CURRENT measured state.
            #
            # Paper:
            #
            # x_ref_i =
            # x_actual + i*dtMPC*vx_des
            #
            # y_ref_i =
            # y_actual + i*dtMPC*vy_des
            #
            # yaw_ref_i =
            # yaw_actual + i*dtMPC*yaw_rate_des
            # =================================================

            X_ref[
                i,
                0
            ] = (

                x_actual_pos

                +

                prediction_time

                *

                v_des_world[
                    0
                ]
            )


            X_ref[
                i,
                1
            ] = (

                y_actual_pos

                +

                prediction_time

                *

                v_des_world[
                    1
                ]
            )


            X_ref[
                i,
                5
            ] = (

                yaw_actual

                +

                prediction_time

                *

                yaw_rate_des
            )


            # =================================================
            # NEAR-ZERO VX
            #
            # Hold persistent position knot.
            # =================================================

            if (
                abs(
                    v_des_world[
                        0
                    ]
                )
                <
                self.velocity_threshold
            ):

                X_ref[
                    i,
                    0
                ] = (

                    x_start

                    +

                    prediction_time

                    *

                    v_des_world[
                        0
                    ]
                )


            # =================================================
            # NEAR-ZERO VY
            # =================================================

            if (
                abs(
                    v_des_world[
                        1
                    ]
                )
                <
                self.velocity_threshold
            ):

                X_ref[
                    i,
                    1
                ] = (

                    y_start

                    +

                    prediction_time

                    *

                    v_des_world[
                        1
                    ]
                )


            # =================================================
            # NEAR-ZERO YAW RATE
            # =================================================

            if (
                abs(
                    yaw_rate_des
                )
                <
                self.yaw_rate_threshold
            ):

                X_ref[
                    i,
                    5
                ] = (

                    yaw_start

                    +

                    prediction_time

                    *

                    yaw_rate_des
                )


            # =================================================
            # z / roll / pitch remain constant from
            # initial_reference.
            #
            # Linear and angular velocity references also
            # remain constant over the horizon.
            # =================================================


        return X_ref


    # ========================================================
    # MAIN UPDATE
    # ========================================================

    def update(
        self,
        x_actual,
        v_des_body,
        yaw_rate_des,
        z_des,
        roll_des=0.0,
        pitch_des=0.0,
    ):
        """
        Update reference generator and return MPC horizon.

        Recommended use:

            call once per MPC/reference update,

        NOT once per MuJoCo physics substep.
        """

        # ====================================================
        # INPUTS
        # ====================================================

        x_actual = self._validate_state(
            x_actual
        )


        v_des_body = _validate_vector(
            v_des_body,
            "v_des_body",
            (3,),
        )


        yaw_rate_des = float(
            yaw_rate_des
        )

        z_des = float(
            z_des
        )

        roll_des = float(
            roll_des
        )

        pitch_des = float(
            pitch_des
        )


        scalar_inputs = np.array(
            [
                yaw_rate_des,
                z_des,
                roll_des,
                pitch_des,
            ],
            dtype=float,
        )


        if not np.all(
            np.isfinite(
                scalar_inputs
            )
        ):

            raise ValueError(
                "Reference command contains NaN or Inf."
            )


        # ====================================================
        # FIRST RUN
        # ====================================================

        if self.first_run:

            self._initialize_from_state(
                x_actual
            )


        # ====================================================
        # CURRENT ORIENTATION
        # ====================================================

        yaw_actual = float(
            x_actual[
                5
            ]
        )


        # ====================================================
        # YAW KNOT PROTECTION
        # ====================================================

        self._apply_yaw_boundary_protection(
            yaw_actual=
                yaw_actual,
        )


        # ====================================================
        # BODY VELOCITY -> WORLD VELOCITY
        # ====================================================

        v_des_world = (
            self.body_velocity_to_world(
                x_actual=
                    x_actual,

                v_des_body=
                    v_des_body,
            )
        )


        # ====================================================
        # PERSISTENT COMMAND UPDATE
        # ====================================================

        self._update_persistent_command(

            v_des_world=
                v_des_world,

            yaw_rate_des=
                yaw_rate_des,

            z_des=
                z_des,
        )


        # ====================================================
        # SAFE HORIZON START
        # ====================================================

        (
            x_start,
            y_start,
            yaw_start,

        ) = self._compute_safe_start(
            x_actual=
                x_actual,
        )


        # ====================================================
        # INITIAL REFERENCE STATE
        # ====================================================

        initial_reference = (
            self._build_initial_reference(

                x_start=
                    x_start,

                y_start=
                    y_start,

                yaw_start=
                    yaw_start,

                z_des=
                    z_des,

                roll_des=
                    roll_des,

                pitch_des=
                    pitch_des,

                v_des_world=
                    v_des_world,

                yaw_rate_des=
                    yaw_rate_des,
            )
        )


        # ====================================================
        # MPC HORIZON
        # ====================================================

        X_ref = (
            self._build_horizon(

                x_actual=
                    x_actual,

                initial_reference=
                    initial_reference,

                v_des_world=
                    v_des_world,

                yaw_rate_des=
                    yaw_rate_des,

                x_start=
                    x_start,

                y_start=
                    y_start,

                yaw_start=
                    yaw_start,
            )
        )


        # ====================================================
        # OUTPUT
        # ====================================================

        output = {

            # MPC trajectory
            "X_ref":
                X_ref.copy(),


            # Flattened form if solver needs one vector
            "X_ref_flat":
                X_ref.reshape(
                    -1
                ).copy(),


            # Velocity command
            "v_des_body":
                v_des_body.copy(),

            "v_des_world":
                v_des_world.copy(),

            "yaw_rate_des":
                float(
                    yaw_rate_des
                ),


            # Attitude / height command
            "z_des":
                float(
                    z_des
                ),

            "roll_des":
                float(
                    roll_des
                ),

            "pitch_des":
                float(
                    pitch_des
                ),


            # Persistent paper-style command
            "world_position_desired":
                self.world_position_desired.copy(),

            "yaw_desired":
                float(
                    self.yaw_desired
                ),


            # Safe starting knot
            "x_start":
                float(
                    x_start
                ),

            "y_start":
                float(
                    y_start
                ),

            "yaw_start":
                float(
                    yaw_start
                ),


            # Timing
            "reference_update_dt":
                float(
                    self.reference_update_dt
                ),

            "dt_mpc":
                float(
                    self.dt_mpc
                ),

            "horizon_length":
                int(
                    self.horizon_length
                ),
        }


        self.last_output = (
            output
        )


        return output


    # ========================================================
    # UPDATE MPC PREDICTION DT
    # ========================================================

    def update_mpc_sampling_time(
        self,
        dt_mpc,
    ):

        self.dt_mpc = (
            _validate_positive_scalar(
                dt_mpc,
                "dt_mpc",
            )
        )


    # ========================================================
    # UPDATE REFERENCE UPDATE DT
    # ========================================================

    def update_reference_sampling_time(
        self,
        reference_update_dt,
    ):

        self.reference_update_dt = (
            _validate_positive_scalar(
                reference_update_dt,
                "reference_update_dt",
            )
        )


    # ========================================================
    # INTERNAL STATE
    # ========================================================

    def get_internal_state(
        self,
    ):

        return {

            "first_run":
                bool(
                    self.first_run
                ),

            "world_position_desired":
                self.world_position_desired.copy(),

            "yaw_desired":
                float(
                    self.yaw_desired
                ),
        }