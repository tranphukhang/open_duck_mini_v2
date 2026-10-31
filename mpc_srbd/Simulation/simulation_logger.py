import numpy as np


# ============================================================
# SIMULATION LOGGER
# ============================================================

class SimulationLogger:
    """
    Logger dùng chung cho các simulation SRBD-MPC.

    Logger lưu dữ liệu theo từng MPC control cycle.

    Các dữ liệu hiện tại gồm:

        time

        x

        u0

        tau

        mpc_compute_time

        qp_solve_time
    """


    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self):

        self.time = []

        self.x = []

        self.u0 = []

        self.tau = []

        self.mpc_compute_time = []

        self.qp_solve_time = []


        # ----------------------------------------------------
        # Indicates whether finalize() has been called.
        # ----------------------------------------------------

        self.is_finalized = False


    # ========================================================
    # LOG ONE MPC CONTROL CYCLE
    # ========================================================

    def log_step(
        self,
        time,
        x,
        u0,
        tau,
        mpc_compute_time,
        qp_solve_time
    ):
        """
        Log data from one MPC control cycle.

        Parameters
        ----------
        time : float

            Current MuJoCo simulation time.


        x : array_like

            Current SRBD state.


        u0 : array_like

            Current MPC optimal input:

                u_k


        tau : array_like

            Current joint torque command.


        mpc_compute_time : float

            Total MPC computation time for the current cycle.


        qp_solve_time : float

            QP solver computation time.
        """


        # ====================================================
        # PREVENT LOGGING AFTER FINALIZE
        # ====================================================

        if self.is_finalized:

            raise RuntimeError(
                "SimulationLogger đã được finalize. "
                "Không thể tiếp tục log dữ liệu."
            )


        # ====================================================
        # INPUT CONVERSION
        # ====================================================

        time = float(
            time
        )


        x = np.asarray(
            x,
            dtype=float
        ).reshape(-1)


        u0 = np.asarray(
            u0,
            dtype=float
        ).reshape(-1)


        tau = np.asarray(
            tau,
            dtype=float
        ).reshape(-1)


        mpc_compute_time = float(
            mpc_compute_time
        )


        qp_solve_time = float(
            qp_solve_time
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        if not np.isfinite(
            time
        ):

            raise ValueError(
                "time không finite."
            )


        if not np.all(
            np.isfinite(
                x
            )
        ):

            raise ValueError(
                "x chứa NaN hoặc Inf."
            )


        if not np.all(
            np.isfinite(
                u0
            )
        ):

            raise ValueError(
                "u0 chứa NaN hoặc Inf."
            )


        if not np.all(
            np.isfinite(
                tau
            )
        ):

            raise ValueError(
                "tau chứa NaN hoặc Inf."
            )


        if not np.isfinite(
            mpc_compute_time
        ):

            raise ValueError(
                "mpc_compute_time không finite."
            )


        if not np.isfinite(
            qp_solve_time
        ):

            raise ValueError(
                "qp_solve_time không finite."
            )


        # ====================================================
        # APPEND LOG
        # ====================================================

        self.time.append(
            time
        )


        self.x.append(
            x.copy()
        )


        self.u0.append(
            u0.copy()
        )


        self.tau.append(
            tau.copy()
        )


        self.mpc_compute_time.append(
            mpc_compute_time
        )


        self.qp_solve_time.append(
            qp_solve_time
        )


    # ========================================================
    # FINALIZE LOG
    # ========================================================

    def finalize(self):
        """
        Convert all log buffers from Python lists
        to NumPy arrays.

        Sau khi finalize(), logger không nhận thêm dữ liệu.
        """


        if self.is_finalized:

            return


        self.time = np.asarray(
            self.time,
            dtype=float
        )


        self.x = np.asarray(
            self.x,
            dtype=float
        )


        self.u0 = np.asarray(
            self.u0,
            dtype=float
        )


        self.tau = np.asarray(
            self.tau,
            dtype=float
        )


        self.mpc_compute_time = np.asarray(
            self.mpc_compute_time,
            dtype=float
        )


        self.qp_solve_time = np.asarray(
            self.qp_solve_time,
            dtype=float
        )


        self.is_finalized = True


    # ========================================================
    # NUMBER OF LOGGED MPC CYCLES
    # ========================================================

    def __len__(self):

        return len(
            self.time
        )


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    def print_summary(self):
        """
        Print basic simulation statistics.

        finalize() sẽ được gọi tự động nếu cần.
        """


        if not self.is_finalized:

            self.finalize()


        print("\n")
        print("=" * 70)
        print("SIMULATION LOG SUMMARY")
        print("=" * 70)


        print(
            "Logged MPC cycles =",
            len(
                self
            )
        )


        # ----------------------------------------------------
        # No logged data
        # ----------------------------------------------------

        if len(
            self
        ) == 0:

            print(
                "Không có dữ liệu simulation được log."
            )

            return


        # ----------------------------------------------------
        # MPC computation time
        # ----------------------------------------------------

        if self.mpc_compute_time.size > 0:

            print(
                "Average MPC compute time =",
                np.mean(
                    self.mpc_compute_time
                ),
                "s"
            )


            print(
                "Maximum MPC compute time =",
                np.max(
                    self.mpc_compute_time
                ),
                "s"
            )


        # ----------------------------------------------------
        # QP solve time
        # ----------------------------------------------------

        if self.qp_solve_time.size > 0:

            print(
                "Average QP solve time =",
                np.mean(
                    self.qp_solve_time
                ),
                "s"
            )


            print(
                "Maximum QP solve time =",
                np.max(
                    self.qp_solve_time
                ),
                "s"
            )


        # ----------------------------------------------------
        # Initial and final SRBD state
        # ----------------------------------------------------

        if (
            self.x.ndim == 2
            and
            self.x.shape[0] > 0
        ):

            print(
                "\nInitial logged state ="
            )


            print(
                self.x[
                    0
                ]
            )


            print(
                "\nFinal logged state ="
            )


            print(
                self.x[
                    -1
                ]
            )