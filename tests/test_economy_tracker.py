import unittest

from economy_tracker import EconomyTracker


class EconomyTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = EconomyTracker(confirmation_reads=2, max_potion_drop=10)

    def confirm(self, **values):
        self.tracker.update(**values)
        self.tracker.update(**values)

    def test_tracks_positive_meso_changes_only(self):
        self.confirm(meso=1000)
        self.confirm(meso=1250)
        self.confirm(meso=1100)
        self.assertEqual(self.tracker.snapshot().meso_gained, 250)

    def test_tracks_hp_and_mp_decreases(self):
        self.confirm(hp_count=80, mp_count=126)
        self.confirm(hp_count=78, mp_count=125)
        snapshot = self.tracker.snapshot(hp_price=100, mp_price=200)
        self.assertEqual(snapshot.hp_consumed, 2)
        self.assertEqual(snapshot.mp_consumed, 1)
        self.assertEqual(snapshot.potion_cost, 400)

    def test_restock_does_not_count_as_consumption(self):
        self.confirm(hp_count=80)
        self.confirm(hp_count=100)
        self.assertEqual(self.tracker.snapshot().hp_consumed, 0)

    def test_rejects_large_ocr_drop(self):
        self.confirm(mp_count=126)
        self.confirm(mp_count=26)
        self.assertEqual(self.tracker.snapshot().mp_consumed, 0)
        self.assertEqual(self.tracker.last_mp_count, 126)

    def test_net_profit_subtracts_manual_prices(self):
        self.confirm(meso=1000, hp_count=10, mp_count=10)
        self.confirm(meso=1600, hp_count=8, mp_count=9)
        snapshot = self.tracker.snapshot(hp_price=100, mp_price=50)
        self.assertEqual(snapshot.net_profit, 350)


if __name__ == "__main__":
    unittest.main()
