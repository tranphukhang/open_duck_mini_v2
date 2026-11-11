
import torch
from contact_scheduler import ContactScheduler


def test_local_time_phase():
    scheduler = ContactScheduler(
        dt=0.02,
        period_phase=0.4,
        shift=0.4
    )

    times = torch.tensor([
        0.00, 0.20, 0.38,
        0.40, 0.42, 0.58,
        0.60, 0.70, 0.78,
        0.80, 1.00
    ])

    expected = torch.tensor([
        0.00, 0.20, 0.38,
        0.00, 0.02, 0.18,
        0.00, 0.10, 0.18,
        0.00, 0.00
    ])

    j = scheduler.step_index(times)
    local = scheduler.local_time_phase(times, j)

    print("Time (s) | Step j | Local (s) | Expected (s)")
    print("-" * 48)

    for t, index, value, exp in zip(times, j, local, expected):
        print(
            f"{t.item():8.2f} | "
            f"{index.item():6d} | "
            f"{value.item():9.5f} | "
            f"{exp.item():12.5f}"
        )

    torch.testing.assert_close(
        local, expected, atol=1e-6, rtol=0
    )

    print("\nTEST local_time_phase: PASS")


if __name__ == "__main__":
    test_local_time_phase()
