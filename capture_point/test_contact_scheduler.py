import torch
from contact_scheduler import ContactScheduler


def test_timeline():
    """Kiem tra toan bo lich tu t=0 den t=1.4 s (dt=0.02 s)."""
    scheduler = ContactScheduler(
        dt=0.02, period_phase=0.4, shift=0.4,
        double_support=0.1, num_envs=1
    )

    # Theo thong so mac dinh: shift=20 tick, mot buoc=10 tick, DSP=2 tick
    shift_ticks = 20
    step_ticks = 10
    dsp_ticks = 2
    last_step = torch.full((1,), -1, dtype=torch.long)
    event_ticks = []
    print('   t  |  j | local |  phase | stance | contact [L,R] | new | adjust')
    print('-' * 76)

    shown_ticks = {0, 19, 20, 21, 22, 28, 29, 30, 31, 32, 38, 39,
                   40, 42, 48, 49, 50, 59, 60, 69, 70}

    for k in range(71):
        t = scheduler.t
        j = scheduler.step_index(t)
        local = scheduler.local_time_phase(t, j)
        shift, ds, ss = scheduler.phase_state(j, local)
        stance = scheduler.stance_feet(j)
        contact = scheduler.contact_state(stance, ss)
        new = scheduler.new_step_event(j, last_step)
        adjust = scheduler.allow_footstep_adjust(t, j)

        # Ket qua mong doi tinh doc lap theo so tick (khong goi lai ham can test)
        if k < shift_ticks:
            expected_j = -1
            expected_local = k * scheduler.dt
            expected_phase = 'SHIFT'
            expected_contact = [True, True]
            expected_adjust = False
            expected_new = False
        else:
            offset = k - shift_ticks
            expected_j = offset // step_ticks
            local_tick = offset % step_ticks
            expected_local = local_tick * scheduler.dt
            expected_phase = 'DSP' if local_tick < dsp_ticks else 'SSP'
            expected_contact = (
                [True, True] if expected_phase == 'DSP'
                else ([True, False] if expected_j % 2 == 0 else [False, True])
            )
            expected_adjust = local_tick != step_ticks - 1
            expected_new = local_tick == 0

        actual_phase = 'SHIFT' if shift.item() else ('DSP' if ds.item() else 'SSP')
        assert j.item() == expected_j, f'j sai tai tick={k}'
        assert abs(local.item() - expected_local) < 1e-6, f'local sai tai tick={k}'
        assert int(shift.item()) + int(ds.item()) + int(ss.item()) == 1, (
            f'Pha chong lap hoac thieu tai tick={k}'
        )
        assert actual_phase == expected_phase, f'phase sai tai tick={k}'
        assert stance.item() == max(expected_j, 0) % 2, f'stance sai tai tick={k}'
        assert contact[0].tolist() == expected_contact, f'contact sai tai tick={k}'
        assert new.item() == expected_new, f'new_step sai tai tick={k}'
        assert adjust.item() == expected_adjust, f'allow_adjust sai tai tick={k}'

        if new.item():
            event_ticks.append(k)

        if k in shown_ticks:
            print(
                f'{t.item():5.2f} | {j.item():2d} | {local.item():5.2f} | '
                f'{actual_phase:>6} | {stance.item():6d} | '
                f'{str(contact[0].tolist()):13} | '
                f'{str(new.item()):5} | {adjust.item()}'
            )

        # Giong thu tu trong walker.py: cap nhat last_step SAU khi xet new
        last_step = torch.where(j >= 0, j, torch.full_like(j, -1))
        if k < 70:
            scheduler.count()

    assert event_ticks == [20, 30, 40, 50, 60, 70]
    print('\nTEST 1 - Full timeline: PASS')


def test_batch():
    """Kiem tra tensor gom nhieu thoi diem / nhieu robot."""
    scheduler = ContactScheduler(num_envs=6)
    t = torch.tensor([0.20, 0.40, 0.44, 0.60, 0.64, 0.78])
    j = scheduler.step_index(t)
    local = scheduler.local_time_phase(t, j)
    shift, ds, ss = scheduler.phase_state(j, local)
    stance = scheduler.stance_feet(j)
    contact = scheduler.contact_state(stance, ss)
    adjust = scheduler.allow_footstep_adjust(t, j)

    assert j.tolist() == [-1, 0, 0, 1, 1, 1]
    assert contact.tolist() == [
        [True, True], [True, True], [True, False],
        [True, True], [False, True], [False, True]
    ]
    assert adjust.tolist() == [False, True, True, True, True, False]
    assert (shift.int() + ds.int() + ss.int()).tolist() == [1] * 6
    print('TEST 2 - Batch schedule: PASS')


def test_reset():
    """Kiem tra reset mot robot va reset tat ca."""
    scheduler = ContactScheduler(num_envs=3)
    for _ in range(7):
        scheduler.count()

    scheduler.reset(env_ids=torch.tensor([1]))
    assert scheduler.tick.tolist() == [7, 0, 7]
    assert torch.allclose(scheduler.t, torch.tensor([0.14, 0.0, 0.14]))

    scheduler.reset()
    assert scheduler.tick.tolist() == [0, 0, 0]
    assert scheduler.t.tolist() == [0.0, 0.0, 0.0]
    print('TEST 3 - Reset: PASS')


if __name__ == '__main__':
    test_timeline()
    test_batch()
    test_reset()
    print('\nALL CONTACT SCHEDULER TESTS PASSED')
