import numpy as np


# ============================================================
# SKEW-SYMMETRIC MATRIX
# ============================================================

def skew_symmetric(vector):
    """
    Tạo ma trận skew-symmetric [r]_x sao cho:

        [r]_x @ F = r x F

    Parameters
    ----------
    vector : array_like, shape (3,)
        Vector 3D.

    Returns
    -------
    skew : np.ndarray, shape (3, 3)
        Ma trận skew-symmetric.
    """

    vector = np.asarray(
        vector,
        dtype=float
    ).reshape(3)


    if not np.all(
        np.isfinite(
            vector
        )
    ):

        raise ValueError(
            "vector chứa NaN hoặc Inf."
        )


    x, y, z = vector


    skew = np.array(
        [
            [0.0, -z,   y],
            [z,    0.0, -x],
            [-y,   x,   0.0]
        ],
        dtype=float
    )


    return skew


# ============================================================
# BUILD ONE CONTINUOUS SRBD MODEL
# ============================================================

def build_continuous_model(
    Ew_inverse,
    mass_CoM,
    W_inertia_CoM,
    W_r_L,
    W_r_R,
    gravity=9.81
):
    """
    Xây dựng một mô hình SRBD liên tục tại một prediction step:

        x_dot = Ac @ x + Bc @ u


    State
    -----

        x =
        [
            W_position_CoM,          # 3
            W_euler_base,            # 3
            W_linear_velocity_CoM,   # 3
            W_angular_velocity_base, # 3
            1                        # 1
        ]


    Input
    -----

        u =
        [
            W_F_L,   # 3
            W_M_L,   # 3
            W_F_R,   # 3
            W_M_R    # 3
        ]


    Moment arms
    -----------

        W_r_L
            =
        W_p_L6 - W_p_CoM


        W_r_R
            =
        W_p_R6 - W_p_CoM


    Angular dynamics
    ----------------

        W_omega_dot
            =
        W_I_CoM^-1
        (
              W_r_L x W_F_L
            + W_M_L
            + W_r_R x W_F_R
            + W_M_R
        )


    Notes
    -----

    Hàm này chỉ xây dựng mô hình tại MỘT prediction step.

    Nó được giữ lại để:

        1. tương thích với implementation hiện tại;

        2. được gọi nhiều lần bởi:

               build_continuous_model_sequence()

           khi xây mô hình LTV trên toàn prediction horizon.


    Parameters
    ----------
    Ew_inverse : np.ndarray, shape (3, 3)

        Ma trận biến đổi:

            euler_dot
                =
            Ew_inverse @ W_omega_base


    mass_CoM : float

        Tổng khối lượng robot [kg].


    W_inertia_CoM : np.ndarray, shape (3, 3)

        Whole-body inertia tensor tại CoM,
        biểu diễn trong World frame [kg.m^2].

        Trong MPC hiện tại giá trị này vẫn được
        freeze tại cấu hình khởi tạo.


    W_r_L : np.ndarray, shape (3,)

        Moment arm từ CoM đến L6:

            W_r_L
                =
            W_p_L6 - W_p_CoM


    W_r_R : np.ndarray, shape (3,)

        Moment arm từ CoM đến R6:

            W_r_R
                =
            W_p_R6 - W_p_CoM


    gravity : float, optional

        Gia tốc trọng trường [m/s^2].

        Default:
            9.81


    Returns
    -------
    Ac : np.ndarray, shape (13, 13)

        Continuous state matrix.


    Bc : np.ndarray, shape (13, 12)

        Continuous input matrix.
    """


    # ========================================================
    # 1. INPUT CONVERSION
    # ========================================================

    Ew_inverse = np.asarray(
        Ew_inverse,
        dtype=float
    )


    W_inertia_CoM = np.asarray(
        W_inertia_CoM,
        dtype=float
    )


    W_r_L = np.asarray(
        W_r_L,
        dtype=float
    ).reshape(-1)


    W_r_R = np.asarray(
        W_r_R,
        dtype=float
    ).reshape(-1)


    mass_CoM = float(
        mass_CoM
    )


    gravity = float(
        gravity
    )


    # ========================================================
    # 2. INPUT VALIDATION
    # ========================================================

    if Ew_inverse.shape != (
        3,
        3
    ):

        raise ValueError(
            "Ew_inverse phải có shape (3, 3)."
        )


    if W_inertia_CoM.shape != (
        3,
        3
    ):

        raise ValueError(
            "W_inertia_CoM phải có shape (3, 3)."
        )


    if W_r_L.shape != (
        3,
    ):

        raise ValueError(
            "W_r_L phải có shape (3,)."
        )


    if W_r_R.shape != (
        3,
    ):

        raise ValueError(
            "W_r_R phải có shape (3,)."
        )


    if mass_CoM <= 0.0:

        raise ValueError(
            "mass_CoM phải lớn hơn 0."
        )


    if gravity <= 0.0:

        raise ValueError(
            "gravity phải lớn hơn 0."
        )


    # --------------------------------------------------------
    # Finite validation
    # --------------------------------------------------------

    for name, value in (
        (
            "Ew_inverse",
            Ew_inverse
        ),
        (
            "W_inertia_CoM",
            W_inertia_CoM
        ),
        (
            "W_r_L",
            W_r_L
        ),
        (
            "W_r_R",
            W_r_R
        )
    ):

        if not np.all(
            np.isfinite(
                value
            )
        ):

            raise ValueError(
                f"{name} chứa NaN hoặc Inf."
            )


    if not np.isfinite(
        mass_CoM
    ):

        raise ValueError(
            "mass_CoM không finite."
        )


    if not np.isfinite(
        gravity
    ):

        raise ValueError(
            "gravity không finite."
        )


    # ========================================================
    # 3. CONSTANT MATRICES
    # ========================================================

    I3 = np.eye(
        3,
        dtype=float
    )


    W_gravity = np.array(
        [
            0.0,
            0.0,
            -gravity
        ],
        dtype=float
    )


    # ========================================================
    # 4. INERTIA INVERSE
    # ========================================================

    try:

        W_inertia_CoM_inverse = (
            np.linalg.inv(
                W_inertia_CoM
            )
        )


    except np.linalg.LinAlgError as exc:

        raise ValueError(
            "W_inertia_CoM không khả nghịch."
        ) from exc


    # ========================================================
    # 5. MOMENT-ARM SKEW MATRICES
    #
    #     [r]_x F
    #
    #         =
    #
    #     r x F
    # ========================================================

    W_r_L_skew = (
        skew_symmetric(
            W_r_L
        )
    )


    W_r_R_skew = (
        skew_symmetric(
            W_r_R
        )
    )


    # ========================================================
    # 6. CONTINUOUS A MATRIX
    #
    # State order:
    #
    #   0:3    W_position_CoM
    #
    #   3:6    W_euler_base
    #
    #   6:9    W_linear_velocity_CoM
    #
    #   9:12   W_angular_velocity_base
    #
    #   12     constant state = 1
    # ========================================================

    Ac = np.zeros(
        (
            13,
            13
        ),
        dtype=float
    )


    # --------------------------------------------------------
    # CoM position dynamics
    #
    #     p_dot = v
    # --------------------------------------------------------

    Ac[
        0:3,
        6:9
    ] = I3


    # --------------------------------------------------------
    # Euler kinematics
    #
    #     euler_dot
    #
    #         =
    #
    #     Ew_inverse @ omega
    # --------------------------------------------------------

    Ac[
        3:6,
        9:12
    ] = Ew_inverse


    # --------------------------------------------------------
    # Linear dynamics gravity term
    #
    #     v_dot
    #
    #         =
    #
    #     ...
    #
    #         +
    #
    #     g
    # --------------------------------------------------------

    Ac[
        6:9,
        12
    ] = W_gravity


    # --------------------------------------------------------
    # Augmented constant state:
    #
    #     d(1)/dt = 0
    #
    # Ac[12, :] remains zero.
    # --------------------------------------------------------


    # ========================================================
    # 7. CONTINUOUS B MATRIX
    #
    # Input order:
    #
    #   0:3    W_F_L
    #
    #   3:6    W_M_L
    #
    #   6:9    W_F_R
    #
    #   9:12   W_M_R
    # ========================================================

    Bc = np.zeros(
        (
            13,
            12
        ),
        dtype=float
    )


    # ========================================================
    # 7.1 LINEAR DYNAMICS
    #
    #     v_dot
    #
    #         =
    #
    #     1/m * W_F_L
    #
    #         +
    #
    #     1/m * W_F_R
    #
    #         +
    #
    #     W_gravity
    # ========================================================

    Bc[
        6:9,
        0:3
    ] = (
        I3
        /
        mass_CoM
    )


    Bc[
        6:9,
        6:9
    ] = (
        I3
        /
        mass_CoM
    )


    # ========================================================
    # 7.2 ANGULAR DYNAMICS
    #
    #     omega_dot
    #
    #         =
    #
    #     I_CoM^-1
    #     (
    #           r_L x F_L
    #         + M_L
    #         + r_R x F_R
    #         + M_R
    #     )
    # ========================================================


    # --------------------------------------------------------
    # Left contact force
    #
    #     I^-1 [r_L]_x F_L
    # --------------------------------------------------------

    Bc[
        9:12,
        0:3
    ] = (
        W_inertia_CoM_inverse
        @
        W_r_L_skew
    )


    # --------------------------------------------------------
    # Left contact moment
    #
    #     I^-1 M_L
    # --------------------------------------------------------

    Bc[
        9:12,
        3:6
    ] = (
        W_inertia_CoM_inverse
    )


    # --------------------------------------------------------
    # Right contact force
    #
    #     I^-1 [r_R]_x F_R
    # --------------------------------------------------------

    Bc[
        9:12,
        6:9
    ] = (
        W_inertia_CoM_inverse
        @
        W_r_R_skew
    )


    # --------------------------------------------------------
    # Right contact moment
    #
    #     I^-1 M_R
    # --------------------------------------------------------

    Bc[
        9:12,
        9:12
    ] = (
        W_inertia_CoM_inverse
    )


    # ========================================================
    # 8. FINAL VALIDATION
    # ========================================================

    if Ac.shape != (
        13,
        13
    ):

        raise RuntimeError(
            "Ac phải có shape (13, 13)."
        )


    if Bc.shape != (
        13,
        12
    ):

        raise RuntimeError(
            "Bc phải có shape (13, 12)."
        )


    if not np.all(
        np.isfinite(
            Ac
        )
    ):

        raise RuntimeError(
            "Ac chứa NaN hoặc Inf."
        )


    if not np.all(
        np.isfinite(
            Bc
        )
    ):

        raise RuntimeError(
            "Bc chứa NaN hoặc Inf."
        )


    return (
        Ac,
        Bc
    )


# ============================================================
# BUILD CONTINUOUS SRBD MODEL SEQUENCE
# ============================================================

def build_continuous_model_sequence(
    Ew_inverse,
    mass_CoM,
    W_inertia_CoM,
    W_r_L_sequence,
    W_r_R_sequence,
    gravity=9.81
):
    """
    Xây dựng chuỗi mô hình SRBD liên tục trên prediction horizon.

    Mô hình tại prediction step j:

        x_dot_j
            =
        Ac_j @ x_j
            +
        Bc_j @ u_j


    Trong implementation hiện tại:

        Ew_inverse

        mass_CoM

        W_inertia_CoM

    vẫn được xem là cố định trên prediction horizon.


    Thành phần thay đổi theo prediction step là:

        W_r_L_sequence[j]

        W_r_R_sequence[j]


    với:

        W_r_L_sequence[j]
            =
        W_p_L6_plan[j]
            -
        W_p_CoM_ref[j]


        W_r_R_sequence[j]
            =
        W_p_R6_plan[j]
            -
        W_p_CoM_ref[j]


    Do đó:

        Bc_j

    có thể thay đổi theo từng prediction step.


    Parameters
    ----------
    Ew_inverse : np.ndarray, shape (3, 3)

        Euler-rate transformation matrix tại MPC update hiện tại.


    mass_CoM : float

        Tổng khối lượng robot [kg].


    W_inertia_CoM : np.ndarray, shape (3, 3)

        Whole-body inertia tensor tại CoM,
        biểu diễn trong World frame.


    W_r_L_sequence : np.ndarray, shape (N, 3)

        Chuỗi nominal left moment arm:

            [
                W_r_L_0
                W_r_L_1
                ...
                W_r_L_N-1
            ]


    W_r_R_sequence : np.ndarray, shape (N, 3)

        Chuỗi nominal right moment arm.


    gravity : float, optional

        Gia tốc trọng trường [m/s^2].


    Returns
    -------
    Ac_sequence : np.ndarray, shape (N, 13, 13)

        Continuous state matrices:

            Ac_sequence[j] = Ac_j


    Bc_sequence : np.ndarray, shape (N, 13, 12)

        Continuous input matrices:

            Bc_sequence[j] = Bc_j
    """


    # ========================================================
    # 1. INPUT CONVERSION
    # ========================================================

    W_r_L_sequence = np.asarray(
        W_r_L_sequence,
        dtype=float
    )


    W_r_R_sequence = np.asarray(
        W_r_R_sequence,
        dtype=float
    )


    # ========================================================
    # 2. SEQUENCE VALIDATION
    # ========================================================

    if W_r_L_sequence.ndim != 2:

        raise ValueError(
            "W_r_L_sequence phải là ma trận 2D "
            "có shape (N, 3)."
        )


    if W_r_R_sequence.ndim != 2:

        raise ValueError(
            "W_r_R_sequence phải là ma trận 2D "
            "có shape (N, 3)."
        )


    if W_r_L_sequence.shape[1] != 3:

        raise ValueError(
            "W_r_L_sequence phải có shape (N, 3)."
        )


    if W_r_R_sequence.shape[1] != 3:

        raise ValueError(
            "W_r_R_sequence phải có shape (N, 3)."
        )


    if (
        W_r_L_sequence.shape
        !=
        W_r_R_sequence.shape
    ):

        raise ValueError(
            "W_r_L_sequence và W_r_R_sequence "
            "phải có cùng shape."
        )


    N = (
        W_r_L_sequence.shape[0]
    )


    if N <= 0:

        raise ValueError(
            "Prediction horizon N phải lớn hơn 0."
        )


    if not np.all(
        np.isfinite(
            W_r_L_sequence
        )
    ):

        raise ValueError(
            "W_r_L_sequence chứa NaN hoặc Inf."
        )


    if not np.all(
        np.isfinite(
            W_r_R_sequence
        )
    ):

        raise ValueError(
            "W_r_R_sequence chứa NaN hoặc Inf."
        )


    # ========================================================
    # 3. ALLOCATE MODEL SEQUENCES
    # ========================================================

    Ac_sequence = np.zeros(
        (
            N,
            13,
            13
        ),
        dtype=float
    )


    Bc_sequence = np.zeros(
        (
            N,
            13,
            12
        ),
        dtype=float
    )


    # ========================================================
    # 4. BUILD MODEL AT EACH PREDICTION STEP
    # ========================================================

    for j in range(
        N
    ):

        W_r_L_j = (
            W_r_L_sequence[
                j,
                :
            ]
        )


        W_r_R_j = (
            W_r_R_sequence[
                j,
                :
            ]
        )


        (
            Ac_j,
            Bc_j

        ) = build_continuous_model(

            Ew_inverse=Ew_inverse,

            mass_CoM=mass_CoM,

            W_inertia_CoM=W_inertia_CoM,

            W_r_L=W_r_L_j,

            W_r_R=W_r_R_j,

            gravity=gravity
        )


        Ac_sequence[
            j,
            :,
            :
        ] = Ac_j


        Bc_sequence[
            j,
            :,
            :
        ] = Bc_j


    # ========================================================
    # 5. FINAL VALIDATION
    # ========================================================

    if Ac_sequence.shape != (
        N,
        13,
        13
    ):

        raise RuntimeError(
            "Ac_sequence sai dimension."
        )


    if Bc_sequence.shape != (
        N,
        13,
        12
    ):

        raise RuntimeError(
            "Bc_sequence sai dimension."
        )


    if not np.all(
        np.isfinite(
            Ac_sequence
        )
    ):

        raise RuntimeError(
            "Ac_sequence chứa NaN hoặc Inf."
        )


    if not np.all(
        np.isfinite(
            Bc_sequence
        )
    ):

        raise RuntimeError(
            "Bc_sequence chứa NaN hoặc Inf."
        )


    return (
        Ac_sequence,
        Bc_sequence
    )