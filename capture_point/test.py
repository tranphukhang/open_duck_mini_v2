
import unittest
import torch

from contact_scheduler import ContactScheduler


class TestContactScheduler(unittest.TestCase):

    # TEST 1: Khoi tao
    def test_initialization(self):
        scheduler = ContactScheduler(dt=0.02, num_envs=2)

        self.assertEqual(scheduler.tick.tolist(), [0, 0])
        self.assertEqual(scheduler.t.tolist(), [0.0, 0.0])

        print("TEST 1: Initialization - PASS")

    # TEST 2: Dem thoi gian
    def test_count(self):
        scheduler = ContactScheduler(dt=0.02)

        for _ in range(10):
            scheduler.count()

        self.assertEqual(scheduler.tick.item(), 10)
        self.assertAlmostEqual(scheduler.t.item(), 0.2, places=6)

        print("TEST 2: Count - PASS")

    # TEST 3: Xac dinh chi so buoc theo thoi gian
    def test_step_index(self):
        scheduler = ContactScheduler(
            dt=0.02,
            period_phase=0.4,
            shift=0.4
        )

        times = torch.tensor([
            0.00, 0.20, 0.38,
            0.40, 0.50, 0.58,
            0.60, 0.70, 0.78,
            0.80, 1.00
        ])

        expected = [-1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 3]

        result = scheduler.step_index(times)

        print("\nTime (s) | Step Index")
        print("---------------------")

        for t, j in zip(times, result):
            print(f"{t.item():8.2f} | {j.item():10d}")

        self.assertEqual(result.tolist(), expected)

        print("TEST 3: Step Index - PASS")

    # TEST 4: Kiem tra cac moc chuyen buoc
    def test_step_boundary(self):
        scheduler = ContactScheduler(dt=0.02)

        expected = {
            0: -1,
            19: -1,
            20: 0,
            29: 0,
            30: 1,
            39: 1,
            40: 2,
            50: 3
        }

        for tick in range(51):
            if tick in expected:
                j = scheduler.step_index(scheduler.t).item()
                self.assertEqual(j, expected[tick])

            if tick < 50:
                scheduler.count()

        print("TEST 4: Step Boundary - PASS")

    # TEST 5: Kiem tra sai so lam tron
    def test_epsilon(self):
        scheduler = ContactScheduler(dt=0.02)

        eps = 1e-3 * scheduler.dt

        times = torch.tensor([
            0.4 - 2 * eps,
            0.4 - eps / 2
        ], dtype=torch.float64)

        result = scheduler.step_index(times)

        self.assertEqual(result.tolist(), [-1, 0])

        print("TEST 5: Epsilon - PASS")

    # TEST 6: Reset
    def test_reset(self):
        scheduler = ContactScheduler(dt=0.02, num_envs=2)

        for _ in range(5):
            scheduler.count()

        scheduler.reset(env_ids=torch.tensor([1]))

        self.assertEqual(scheduler.tick.tolist(), [5, 0])
        self.assertAlmostEqual(scheduler.t[0].item(), 0.1, places=6)
        self.assertEqual(scheduler.t[1].item(), 0.0)

        scheduler.reset()

        self.assertEqual(scheduler.tick.tolist(), [0, 0])
        self.assertEqual(scheduler.t.tolist(), [0.0, 0.0])

        print("TEST 6: Reset - PASS")

    # TEST 7: Thay doi tham so
    def test_custom_parameters(self):
        scheduler = ContactScheduler(
            dt=0.02,
            period_phase=0.6,
            shift=0.2
        )

        times = torch.tensor([
            0.0, 0.2, 0.49,
            0.5, 0.79, 0.8
        ], dtype=torch.float64)

        expected = [-1, 0, 0, 1, 1, 2]

        result = scheduler.step_index(times)

        self.assertEqual(result.tolist(), expected)

        print("TEST 7: Custom Parameters - PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
