import unittest

from economy_tracker import EconomyTracker


class EconomyTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = EconomyTracker(confirmation_reads=2, max_potion_drop=5)

    def confirm(self, **values):
        self.tracker.update(**values)
        self.tracker.update(**values)

    def capture_start(self, hp=80, mp=100):
        self.confirm(hp_count=hp, mp_count=mp)
        self.assertTrue(self.tracker.has_potion_start)
        self.assertEqual(self.tracker.potion_phase, "ready")

    def test_tracks_net_meso_change_from_reset_baseline(self):
        self.confirm(meso=1000)
        self.confirm(meso=1250)
        self.confirm(meso=1100)
        self.assertEqual(self.tracker.snapshot().meso_gained, 100)

    def test_meso_ocr_drop_and_recovery_does_not_inflate_income(self):
        self.confirm(meso=1_043_338)
        self.confirm(meso=43_338)
        self.confirm(meso=1_043_338)
        self.assertEqual(self.tracker.snapshot().meso_gained, 0)

    def test_meso_net_change_can_be_negative(self):
        self.confirm(meso=1_000)
        self.confirm(meso=750)
        self.assertEqual(self.tracker.snapshot().meso_gained, -250)

    def test_first_confirmed_counts_become_start_snapshot(self):
        self.capture_start(hp=1510, mp=773)
        self.assertEqual(self.tracker.initial_hp_count, 1510)
        self.assertEqual(self.tracker.initial_mp_count, 773)
        self.assertEqual(self.tracker.snapshot().hp_consumed, 0)
        self.assertEqual(self.tracker.snapshot().mp_consumed, 0)

    def test_intermediate_training_reads_are_ignored(self):
        self.capture_start(hp=1510, mp=773)
        self.confirm(hp_count=1500, mp_count=763)
        self.confirm(hp_count=1508, mp_count=772)

        self.assertEqual(self.tracker.last_hp_count, 1510)
        self.assertEqual(self.tracker.last_mp_count, 773)
        self.assertEqual(self.tracker.snapshot().hp_consumed, 0)
        self.assertEqual(self.tracker.snapshot().mp_consumed, 0)

    def test_second_ocr_snapshot_calculates_consumption(self):
        self.capture_start(hp=1510, mp=773)
        self.assertTrue(self.tracker.begin_potion_settlement())
        self.confirm(hp_count=1508, mp_count=770)

        snapshot = self.tracker.snapshot()
        self.assertEqual(self.tracker.potion_phase, "settled")
        self.assertEqual(snapshot.hp_consumed, 2)
        self.assertEqual(snapshot.mp_consumed, 3)

    def test_large_legitimate_session_consumption_is_allowed(self):
        self.capture_start(hp=1510, mp=773)
        self.tracker.begin_potion_settlement()
        self.confirm(hp_count=900, mp_count=300)

        self.assertEqual(self.tracker.snapshot().hp_consumed, 610)
        self.assertEqual(self.tracker.snapshot().mp_consumed, 473)

    def test_manual_final_counts_calculate_consumption(self):
        self.capture_start(hp=80, mp=100)
        self.assertTrue(self.tracker.settle_potions(72, 91, source="manual"))

        snapshot = self.tracker.snapshot(hp_price=100, mp_price=200)
        self.assertEqual(snapshot.hp_consumed, 8)
        self.assertEqual(snapshot.mp_consumed, 9)
        self.assertEqual(snapshot.potion_cost, 2600)

    def test_manual_settlement_requires_start_snapshot(self):
        self.assertFalse(self.tracker.settle_potions(10, 20))
        self.assertFalse(self.tracker.begin_potion_settlement())

    def test_manual_start_can_replace_wrong_ocr_snapshot(self):
        self.capture_start(hp=7022, mp=2732)
        self.assertTrue(self.tracker.set_potion_start(702, 273, source="manual"))

        self.assertEqual(self.tracker.initial_hp_count, 702)
        self.assertEqual(self.tracker.initial_mp_count, 273)
        self.assertEqual(self.tracker.potion_phase, "ready")

        self.tracker.settle_potions(692, 268)
        snapshot = self.tracker.snapshot()
        self.assertEqual(snapshot.hp_consumed, 10)
        self.assertEqual(snapshot.mp_consumed, 5)

    def test_correcting_start_clears_previous_settlement(self):
        self.capture_start(hp=80, mp=100)
        self.tracker.settle_potions(75, 96)
        self.tracker.set_potion_start(79, 99)

        self.assertIsNone(self.tracker.final_hp_count)
        self.assertIsNone(self.tracker.final_mp_count)
        self.assertEqual(self.tracker.snapshot().hp_consumed, 0)
        self.assertEqual(self.tracker.snapshot().mp_consumed, 0)

    def test_pickups_can_make_snapshot_consumption_negative(self):
        self.capture_start(hp=80, mp=100)
        self.tracker.settle_potions(82, 101)

        snapshot = self.tracker.snapshot(hp_price=100, mp_price=50)
        self.assertEqual(snapshot.hp_consumed, -2)
        self.assertEqual(snapshot.mp_consumed, -1)
        self.assertEqual(snapshot.potion_cost, -250)

    def test_net_profit_uses_settled_snapshot_cost(self):
        self.confirm(meso=1000, hp_count=80, mp_count=100)
        self.confirm(meso=1600)
        self.tracker.settle_potions(78, 99)

        snapshot = self.tracker.snapshot(hp_price=100, mp_price=50)
        self.assertEqual(snapshot.meso_gained, 600)
        self.assertEqual(snapshot.net_profit, 350)

    def test_settlement_can_be_repeated_from_same_start(self):
        self.capture_start(hp=80, mp=100)
        self.tracker.settle_potions(78, 99)
        self.assertEqual(self.tracker.snapshot().hp_consumed, 2)

        self.assertTrue(self.tracker.begin_potion_settlement())
        self.confirm(hp_count=75, mp_count=96)
        self.assertEqual(self.tracker.snapshot().hp_consumed, 5)
        self.assertEqual(self.tracker.snapshot().mp_consumed, 4)

    def test_reset_starts_a_new_snapshot_session(self):
        self.capture_start(hp=80, mp=100)
        self.tracker.settle_potions(75, 96)
        self.tracker.reset()

        self.assertEqual(self.tracker.potion_phase, "start")
        self.assertIsNone(self.tracker.initial_hp_count)
        self.assertIsNone(self.tracker.final_hp_count)
        self.assertEqual(self.tracker.snapshot().hp_consumed, 0)


if __name__ == "__main__":
    unittest.main()
