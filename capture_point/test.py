
import torch
from contact_scheduler import ContactScheduler


def test_contact_state():
    scheduler = ContactScheduler(num_envs=6)

    # 0 = chan trai, 1 = chan phai
    stance = torch.tensor([0, 0, 0, 1, 1, 1])
    ss_condition = torch.tensor([
        False, False, True, False, True, True
    ])

    # Contact = [chan trai, chan phai]
    expected = torch.tensor([
        [True, True],    # Initial Shift
        [True, True],    # DSP, stance trai
        [True, False],   # SSP, stance trai
        [True, True],    # DSP, stance phai
        [False, True],   # SSP, stance phai
        [False, True]    # SSP, stance phai
    ])

    # Goi ham can kiem tra
    result = scheduler.contact_state(stance, ss_condition)

    print("Stance | SSP | Contact [L, R] | Result")
    print("-" * 48)

    for i in range(len(stance)):
        status = (
            "PASS"
            if torch.equal(result[i], expected[i])
            else "FAIL"
        )

        print(
            f"{stance[i].item():6d} | "
            f"{str(ss_condition[i].item()):5s} | "
            f"{str(result[i].tolist()):17s} | "
            f"{status}"
        )

    assert result.shape == (6, 2)
    assert result.dtype == torch.bool
    assert torch.equal(result, expected), "Contact State FAIL"

    print("\nTEST contact_state: PASS")


if __name__ == "__main__":
    test_contact_state()
