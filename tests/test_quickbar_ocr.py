import unittest

import numpy as np

from ui_ocr_extractor import UiOcrExtractor


class StubOcrModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def recognize(self, _image):
        return next(self.outputs)


class QuickbarOcrTests(unittest.TestCase):
    @staticmethod
    def extractor_with_outputs(outputs):
        extractor = UiOcrExtractor.__new__(UiOcrExtractor)
        extractor._model = StubOcrModel(outputs)
        return extractor

    def setUp(self):
        self.image = np.zeros((16, 33, 3), dtype=np.uint8)

    def test_initial_read_prefers_supported_full_length_number(self):
        extractor = self.extractor_with_outputs([
            ("1", 0.99),
            ("1517", 0.88),
            ("1517", 0.79),
            ("157", 0.91),
            ("15", 0.95),
            ("151", 0.82),
        ])

        value, confidence, _ = extractor._recognize_quickbar_digits(
            self.image,
            previous_value=None,
        )

        self.assertEqual(value, 1517)
        self.assertEqual(confidence, 0.88)

    def test_previous_value_rejects_blue_potion_jump_of_ten(self):
        extractor = self.extractor_with_outputs([
            ("763", 0.99),
            ("772", 0.86),
            ("73", 0.98),
            ("772", 0.82),
            ("763", 0.97),
            ("72", 0.96),
        ])

        value, _, _ = extractor._recognize_quickbar_digits(
            self.image,
            previous_value=773,
        )

        self.assertEqual(value, 772)

    def test_previous_value_rejects_false_1500_baseline(self):
        extractor = self.extractor_with_outputs([
            ("1500", 0.99),
            ("1508", 0.85),
            ("1508", 0.82),
            ("158", 0.96),
            ("1500", 0.98),
            ("1508", 0.80),
        ])

        value, _, _ = extractor._recognize_quickbar_digits(
            self.image,
            previous_value=1510,
        )

        self.assertEqual(value, 1508)


if __name__ == "__main__":
    unittest.main()

