
import torch
from contact_scheduler import ContactScheduler


def test_new_step_event():
    scheduler = ContactScheduler()

    # TEST 1: Kiem tra cac truong hop
    j_now = torch.tensor([-1, -1, 0, 0, 1, 1, 2, 3])
    last_step = torch.tensor([-3, -1, -1, 0, 0, 1, 1, 2])

    expected = torch.tensor([
        False, False, True, False,
        True, False, True, True
    ])

    result = scheduler.new_step_event(j_now, last_step)

    print("TEST 1: New Step Event")
    print("j_now | last_step | new_step")
    print("-" * 30)

    for j, last, new in zip(j_now, last_step, result):
        print(
            f"{j.item():5d} | "
            f"{last.item():9d} | "
            f"{str(new.item()):8s}"
        )

    assert torch.equal(result, expected)
    print("TEST 1: PASS\n")

    # TEST 2: Kiem tra theo chu ky dieu khien 0.02 s
    scheduler.reset()
    last_step = torch.tensor([-3])

    print("TEST 2: Time Sequence")
    print("Time (s) | j_now | New Step")
    print("-" * 34)

    detected_ticks = []

    for k in range(45):
        j = scheduler.step_index(scheduler.t)
        new = scheduler.new_step_event(j, last_step)

        if new.item():
            detected_ticks.append(k)
            print(
                f"{scheduler.t.item():8.2f} | "
                f"{j.item():5d} | "
                f"{str(new.item()):8s}"
            )

        # Cap nhat last_step sau khi kiem tra su kien
        last_step = torch.where(
            j >= 0, j, torch.full_like(j, -1)
        )

        scheduler.count()

    # Su kien tai 0.4, 0.6, 0.8 s
    assert detected_ticks == [20, 30, 40]

    print("TEST 2: PASS")
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    test_new_step_event()
