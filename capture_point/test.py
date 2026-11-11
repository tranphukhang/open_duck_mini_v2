
import torch
from contact_scheduler import ContactScheduler


def test_phase_state():
    scheduler = ContactScheduler(
        dt=0.02,
        period_phase=0.4,
        shift=0.4,
        double_support=0.1
    )

    # Du lieu dau vao
    j = torch.tensor([-1, -1, 0, 0, 0, 0, 1, 1])

    local_time = torch.tensor([
        0.00, 0.20,
        0.00, 0.02, 0.039, 0.041,
        0.00, 0.18
    ])

    # Ket qua mong doi
    expected_shift = torch.tensor(
        [1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool
    )
    expected_ds = torch.tensor(
        [0, 0, 1, 1, 1, 0, 1, 0], dtype=torch.bool
    )
    expected_ss = torch.tensor(
        [0, 0, 0, 0, 0, 1, 0, 1], dtype=torch.bool
    )

    # Goi ham can kiem tra
    shift, ds, ss = scheduler.phase_state(j, local_time)

    print(" j | Local (s) | Shift | DSP | Swing")
    print("-" * 40)

    for i in range(len(j)):
        print(
            f"{j[i].item():2d} | "
            f"{local_time[i].item():9.3f} | "
            f"{int(shift[i]):5d} | "
            f"{int(ds[i]):3d} | "
            f"{int(ss[i]):5d}"
        )

    assert torch.equal(shift, expected_shift), "Shift FAIL"
    assert torch.equal(ds, expected_ds), "DSP FAIL"
    assert torch.equal(ss, expected_ss), "Swing FAIL"

    print("\nTEST phase_state: PASS")

    # Kiem tra them sai so so hoc tai ranh gioi DSP/Swing
    boundary_j = torch.tensor([0])
    local_boundary = torch.tensor([0.44]) - 0.4

    _, ds_boundary, ss_boundary = scheduler.phase_state(
        boundary_j, local_boundary
    )

    print("\nBoundary check at t = 0.44 s")
    print(f"Local time: {local_boundary.item():.10f}")
    print(f"DSP: {ds_boundary.item()}")
    print(f"Swing: {ss_boundary.item()}")

    if ds_boundary.item():
        print("WARNING: Float32 rounding affects the boundary.")


if __name__ == "__main__":
    test_phase_state()
