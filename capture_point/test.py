import torch
from contact_scheduler import ContactScheduler


def test_stance_feet():
    scheduler = ContactScheduler()

    # -1: initial shift, duoc quy uoc ve chan trai (j = 0)
    # Chan trai = 0; chan phai = 1
    j = torch.tensor([-1, 0, 1, 2, 3, 4, 5, 6], dtype=torch.long)
    expected = torch.tensor([0, 0, 1, 0, 1, 0, 1, 0], dtype=torch.long)

    result = scheduler.stance_feet(j)

    print("Step j | Stance | Expected | Result")
    print("-------------------------------------")
    for step, actual, target in zip(j.tolist(), result.tolist(), expected.tolist()):
        status = "PASS" if actual == target else "FAIL"
        print(f"{step:6d} | {actual:6d} | {target:8d} | {status}")

    assert torch.equal(result, expected), (
        f"stance_feet() FAIL: got {result.tolist()}, expected {expected.tolist()}"
    )
    print("\nTEST stance_feet: PASS")


if __name__ == "__main__":
    test_stance_feet()
