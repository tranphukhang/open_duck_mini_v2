"""Kiem thu ham reset() cua FootstepPlanner (khong can MuJoCo)."""

import torch
from footstep_planner import FootstepPlanner


def check_close(actual, expected, label):
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
    print(f"  PASS - {label}")


def test_single_robot():
    print("\nTEST 1 - Reset mot robot")
    planner = FootstepPlanner(num_envs=1, device="cpu")
    env_ids = torch.tensor([0], dtype=torch.long)

    # Vi tri hai foot site do FK tra ve, trong he {P}: [Left, Right]
    feet_pos = torch.tensor([
        [[0.01, 0.06, 0.005], [0.02, -0.06, 0.004]],
    ], dtype=torch.float32)

    planner.reset(env_ids, feet_pos)

    check_close(planner.feet0[env_ids], feet_pos, "feet0 = feet_pos")
    check_close(planner.feet_start[env_ids], feet_pos, "feet_start = feet_pos")
    check_close(planner.nominal_landing[env_ids], feet_pos[:, 1, :],
                "nominal_landing = vi tri chan phai")
    check_close(planner.updated_landing[env_ids], feet_pos[:, 1, :],
                "updated_landing = vi tri chan phai")


def test_selected_environments():
    print("\nTEST 2 - Reset chon loc trong 4 robot")
    planner = FootstepPlanner(num_envs=4, device="cpu")
    env_ids = torch.tensor([1, 3], dtype=torch.long)
    feet_pos = torch.tensor([
        [[0.10, 0.07, 0.01], [0.11, -0.07, 0.02]],  # Robot 1
        [[0.30, 0.08, 0.03], [0.31, -0.08, 0.04]],  # Robot 3
    ], dtype=torch.float32)

    planner.reset(env_ids, feet_pos)
    check_close(planner.feet0[env_ids], feet_pos, "feet0 cua robot 1 va 3")
    check_close(planner.feet_start[env_ids], feet_pos, "feet_start cua robot 1 va 3")
    check_close(planner.nominal_landing[env_ids], feet_pos[:, 1, :],
                "nominal_landing cua robot 1 va 3")
    check_close(planner.updated_landing[env_ids], feet_pos[:, 1, :],
                "updated_landing cua robot 1 va 3")

    untouched = torch.tensor([0, 2], dtype=torch.long)
    for name in ("feet0", "feet_start", "nominal_landing", "updated_landing"):
        value = getattr(planner, name)[untouched]
        check_close(value, torch.zeros_like(value), f"{name}: robot 0 va 2 khong doi")


def test_reset_again():
    print("\nTEST 3 - Reset lai sau khi du lieu da thay doi")
    planner = FootstepPlanner(num_envs=2, device="cpu")
    env_ids = torch.tensor([0], dtype=torch.long)
    initial_pos = torch.tensor([
        [[0.00, 0.06, 0.00], [0.00, -0.06, 0.00]],
    ], dtype=torch.float32)
    planner.reset(env_ids, initial_pos)

    # Gia lap trang thai thay doi trong luc walking
    planner.feet_start[0] += 0.2
    planner.nominal_landing[0] += 0.3
    planner.updated_landing[0] += 0.4

    # FK tai thoi diem bat dau episode moi
    new_pos = torch.tensor([
        [[0.50, 0.07, 0.01], [0.52, -0.07, 0.02]],
    ], dtype=torch.float32)
    planner.reset(env_ids, new_pos)

    check_close(planner.feet0[env_ids], new_pos, "feet0 duoc khoi tao lai")
    check_close(planner.feet_start[env_ids], new_pos, "feet_start duoc khoi tao lai")
    check_close(planner.nominal_landing[env_ids], new_pos[:, 1, :],
                "nominal_landing duoc khoi tao lai")
    check_close(planner.updated_landing[env_ids], new_pos[:, 1, :],
                "updated_landing duoc khoi tao lai")

    # Robot 1 khong duoc reset
    for name in ("feet0", "feet_start", "nominal_landing", "updated_landing"):
        value = getattr(planner, name)[1]
        check_close(value, torch.zeros_like(value), f"{name}: robot 1 khong doi")


if __name__ == "__main__":
    test_single_robot()
    test_selected_environments()
    test_reset_again()
    print("\nALL FOOTSTEP PLANNER RESET TESTS PASSED")
