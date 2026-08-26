import unittest

from economy_tracker import EconomyTracker


class EconomyTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = EconomyTracker(confirmation_reads=2, max_potion_drop=10)

    def confirm(self, **values):
        self.tracker.update(**values)
        self.tracker.update(**values)

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

    def test_tracks_hp_and_mp_decreases(self):
        self.confirm(hp_count=80, mp_count=126)
        self.confirm(hp_count=78, mp_count=125)
        snapshot = self.tracker.snapshot(hp_price=100, mp_price=200)
        self.assertEqual(snapshot.hp_consumed, 2)
        self.assertEqual(snapshot.mp_consumed, 1)
        self.assertEqual(snapshot.potion_cost, 400)

    def test_obtained_potions_reduce_accumulated_consumption(self):
        self.confirm(hp_count=80)
        self.confirm(hp_count=75)
        self.confirm(hp_count=77)
        self.assertEqual(self.tracker.snapshot().hp_consumed, 3)

    def test_obtained_potions_can_make_net_consumption_negative(self):
        self.confirm(hp_count=80, mp_count=100)
        self.confirm(hp_count=79, mp_count=98)
        self.confirm(hp_count=88, mp_count=101)
        snapshot = self.tracker.snapshot()
        self.assertEqual(snapshot.hp_consumed, -8)
        self.assertEqual(snapshot.mp_consumed, -1)

    def test_negative_consumption_counts_as_potion_value_gained(self):
        self.confirm(meso=1000, hp_count=80, mp_count=100)
        self.confirm(meso=1200, hp_count=82, mp_count=101)
        snapshot = self.tracker.snapshot(hp_price=100, mp_price=50)
        self.assertEqual(snapshot.potion_cost, -250)
        self.assertEqual(snapshot.net_profit, 450)

    def test_obtained_potions_update_net_cost(self):
        self.confirm(hp_count=80, mp_count=100)
        self.confirm(hp_count=75, mp_count=96)
        self.confirm(hp_count=77, mp_count=97)
        snapshot = self.tracker.snapshot(hp_price=100, mp_price=200)
        self.assertEqual(snapshot.hp_consumed, 3)
        self.assertEqual(snapshot.mp_consumed, 3)
        self.assertEqual(snapshot.potion_cost, 900)

    def test_large_ocr_drop_keeps_last_valid_baseline(self):
        self.confirm(mp_count=126)
        self.confirm(mp_count=26)
        self.assertEqual(self.tracker.snapshot().mp_consumed, 0)
        self.assertEqual(self.tracker.last_mp_count, 126)
        self.confirm(mp_count=125)
        self.assertEqual(self.tracker.snapshot().mp_consumed, 1)

    def test_large_ocr_gain_keeps_last_valid_baseline(self):
        self.confirm(mp_count=2)
        self.confirm(mp_count=12160)
        self.assertEqual(self.tracker.snapshot().mp_consumed, 0)
        self.assertEqual(self.tracker.last_mp_count, 2)
        self.confirm(mp_count=1)
        self.assertEqual(self.tracker.snapshot().mp_consumed, 1)

    def test_blue_potion_ocr_jump_of_ten_is_ignored(self):
        self.confirm(mp_count=773)
        self.confirm(mp_count=763)
        self.assertEqual(self.tracker.last_mp_count, 773)
        self.assertEqual(self.tracker.snapshot().mp_consumed, 0)

        self.confirm(mp_count=772)
        self.assertEqual(self.tracker.snapshot().mp_consumed, 1)

    def test_white_potion_recovery_does_not_become_negative_eight(self):
        self.confirm(hp_count=1510)
        self.confirm(hp_count=1500)
        self.assertEqual(self.tracker.last_hp_count, 1510)
        self.assertEqual(self.tracker.snapshot().hp_consumed, 0)

        self.confirm(hp_count=1508)
        self.assertEqual(self.tracker.snapshot().hp_consumed, 2)

    def test_net_profit_subtracts_manual_prices(self):
        self.confirm(meso=1000, hp_count=10, mp_count=10)
        self.confirm(meso=1600, hp_count=8, mp_count=9)
        snapshot = self.tracker.snapshot(hp_price=100, mp_price=50)
        self.assertEqual(snapshot.net_profit, 350)


if __name__ == "__main__":
    unittest.main()

