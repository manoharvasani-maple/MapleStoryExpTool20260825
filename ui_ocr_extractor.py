import re
import time

import cv2
import numpy as np
import onnxruntime as ort
import yaml

from diagnostics import get_logger
from helper import get_resource_path


logger = get_logger("ocr")

# Bounding box offsets (lv template top-left corner as origin)
LV_OFFSET = (32, 11, 76, 23)
HP_OFFSET = (236, 2, 325, 14)
MP_OFFSET = (347, 2, 433, 14)
EXP_OFFSET = (464, 2, 562, 14)

TEMPLATE_DIST = 573.0

# The two auxiliary templates were cropped from the screenshots supplied for
# this version. Offsets below are relative to each widget's top-left corner.
INVENTORY_TITLE_TEMPLATE_ORIGIN = (8, 7)
INVENTORY_MESO_OFFSET = (66, 305, 128, 323)
QUICKBAR_BOTTOM_TEMPLATE_ORIGIN = (0, 75)
# Include the complete pixel-font counters.  The previous boxes clipped the
# first/last digit edges (especially four-digit HP stacks) and included too
# much of the hotkey label, which could turn 1510 into 1500 or 773 into 763.
QUICKBAR_MP_OFFSET = (40, 29, 70, 45)  # Ins slot, quantity only
# HP's white counter is only eight pixels tall. Keep a slightly larger source
# region and evaluate several stable text bands inside it; one fixed crop works
# for some digits but clips strokes from others (notably 7, 0, and 9).
QUICKBAR_HP_OFFSET = (39, 63, 73, 78)  # Del slot, quantity source region
AUX_TEMPLATE_THRESHOLD = 0.80
AUX_DETECT_INTERVAL_FRAMES = 5

MODEL_DIR = get_resource_path("resources/models/PP-OCRv6_tiny_rec_onnx")
MODEL_PATH = MODEL_DIR / "inference.onnx"
CONFIG_PATH = MODEL_DIR / "inference.yml"


class UiOcrExtractor:
    def __init__(self):
        self._template_a = self._load_template('resources/templates/lv.png')
        self._template_b = self._load_template('resources/templates/shop.png')
        self._inventory_template = self._load_template('resources/templates/inventory_title.png')
        self._quickbar_template = self._load_template('resources/templates/quickbar_bottom.png')

        self._screenshot = np.array([])
        self._size = (0, 0)
        self._scale = 0

        # Top-Left and Bottom-Right corners : [x1, y1, x2, y2]
        self._lv_box = (0, 0, 1, 1)
        self._exp_box = (0, 0, 1, 1)
        self._meso_box = (0, 0, 1, 1)
        self._hp_potion_box = (0, 0, 1, 1)
        self._mp_potion_box = (0, 0, 1, 1)
        self._inventory_available = False
        self._quickbar_available = False
        self._aux_frame_counter = 0
        self._last_aux_state = None
        self._last_log_times = {}

        # Skip check: skip inferencing if no change
        self._lv_last_array = None
        self._exp_last_array = None
        self._lv_last = 0
        self._exp_last = 0
        self._meso_last_array = None
        self._hp_potion_last_array = None
        self._mp_potion_last_array = None
        self._meso_last = None
        self._hp_potion_last = None
        self._mp_potion_last = None

        self._model = PPOCRv6TinyTextRecognition(MODEL_PATH, CONFIG_PATH)

    def reset_economy_cache(self) -> None:
        """Forget cached inventory/quickbar OCR values after a user reset.

        The tracker and OCR cache must start from the same frame.  Otherwise a
        bad quickbar candidate retained by the extractor can immediately become
        the new tracker baseline and keep a potion counter stuck.
        """
        self._meso_last_array = None
        self._hp_potion_last_array = None
        self._mp_potion_last_array = None
        self._meso_last = None
        self._hp_potion_last = None
        self._mp_potion_last = None

    def _should_log(self, key: str, interval_sec: float = 10.0) -> bool:
        now = time.monotonic()
        previous = self._last_log_times.get(key, 0.0)
        if now - previous < interval_sec:
            return False
        self._last_log_times[key] = now
        return True

    def is_available(self) -> bool:
        return self._scale > 0

    def _safe_crop(self, box):
        """
        Clamp a bounding box to the current screenshot bounds and return the
        crop, or None if the screenshot / region is not usable. This protects
        every caller against stale boxes, zero-sized screenshots, or a box
        that fell outside the frame after a resolution/UI change.
        """
        screenshot = self._screenshot
        if screenshot is None or screenshot.size == 0:
            return None

        h, w = screenshot.shape[0], screenshot.shape[1]
        if h <= 0 or w <= 0:
            return None

        x1, y1, x2, y2 = box
        x1 = max(0, min(int(x1), w))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h))
        y2 = max(0, min(int(y2), h))

        if x2 - x1 <= 0 or y2 - y1 <= 0:
            return None

        return screenshot[y1:y2, x1:x2, :3]

    def get_player_level(self) -> tuple:
        # No UI match / bad geometry -> nothing to read yet.
        if not self.is_available():
            return 0, 0

        img = self._safe_crop(self._lv_box)
        if img is None:
            return 0, 0

        # Skip check
        if np.array_equal(self._lv_last_array, img):
            return self._lv_last, 1

        # Inference
        try:
            text, confidence = self._model.recognize(img)
        except Exception:
            if self._should_log("lv_recognition"):
                logger.exception("Level recognition failed crop_shape=%s", img.shape)
            return 0, 0

        # Remove non-digit
        digits = re.sub(r'\D', '', text)
        if (confidence is None or confidence < 0.8 or not digits) and self._should_log(
                "lv_low_confidence"
        ):
            logger.warning(
                "Low-confidence level OCR confidence=%s raw_text=%r crop_shape=%s",
                confidence,
                text,
                img.shape,
            )

        # Target 1 to 3 digits
        if len(digits) >= 1:
            # Take up to 3 digits from the end/main sequence
            level = digits[-3:] if len(digits) >= 3 else digits

            try:
                level_int = int(level)
            except ValueError:
                return 0, 0

            self._lv_last = level_int
            self._lv_last_array = img.copy()
            return level_int, confidence

        return 0, 0

    def get_meso(self) -> tuple:
        if not self._inventory_available:
            return None, None
        return self._get_digit_value(
            self._meso_box,
            "_meso_last_array",
            "_meso_last",
        )

    def get_hp_potion_count(self) -> tuple:
        if not self._quickbar_available:
            return None, None
        return self._get_digit_value(
            self._hp_potion_box,
            "_hp_potion_last_array",
            "_hp_potion_last",
            quickbar_mode="black_on_white",
        )

    def get_mp_potion_count(self) -> tuple:
        if not self._quickbar_available:
            return None, None
        return self._get_digit_value(
            self._mp_potion_box,
            "_mp_potion_last_array",
            "_mp_potion_last",
            quickbar_mode="white_on_black",
        )

    def _get_digit_value(
            self,
            box,
            array_attr: str,
            value_attr: str,
            quickbar_mode=None,
    ) -> tuple:
        img = self._safe_crop(box)
        if img is None:
            return None, None

        last_array = getattr(self, array_attr)
        last_value = getattr(self, value_attr)
        if last_value is not None and np.array_equal(last_array, img):
            return last_value, 1.0

        try:
            if quickbar_mode:
                value, confidence, text = self._recognize_quickbar_digits(
                    img,
                    last_value,
                    quickbar_mode,
                )
            else:
                text, confidence = self._model.recognize(img)
                digits = re.sub(r'\D', '', text)
                value = int(digits) if digits else None
        except Exception:
            if self._should_log(f"numeric_recognition:{value_attr}"):
                logger.exception(
                    "Numeric recognition failed field=%s crop_shape=%s quickbar_mode=%s",
                    value_attr,
                    img.shape,
                    quickbar_mode,
                )
            return None, 0.0

        digits = re.sub(r'\D', '', text or '')
        if (confidence is None or confidence < 0.8 or not digits) and self._should_log(
                f"numeric_low_confidence:{value_attr}"
        ):
            logger.warning(
                "Low-confidence numeric OCR field=%s confidence=%s raw_text=%r crop_shape=%s",
                value_attr,
                confidence,
                text,
                img.shape,
            )
        if not digits:
            return None, 0.0

        if value is None:
            return None, 0.0

        if quickbar_mode and value != last_value:
            logger.info(
                "Quickbar OCR candidate field=%s previous=%s value=%s confidence=%.3f raw_text=%r",
                value_attr,
                last_value,
                value,
                float(confidence or 0.0),
                text,
            )

        # Cache only a successful result. Caching before inference could turn a
        # failed read into a stale value with confidence 1 on the next frame.
        setattr(self, array_attr, img.copy())
        setattr(self, value_attr, value)
        return value, confidence

    def _recognize_quickbar_digits(self, image, previous_value, quickbar_mode=None):
        """Recognize a tiny outlined counter using several conservative views.

        The general OCR model is prone to returning an over-confident partial
        number for MapleStory's 8-pixel counter font.  Enlarging the raw crop
        with both nearest-neighbour and Lanczos interpolation preserves
        different digit details.  We then prefer a full-length consensus and,
        once a baseline exists, candidates close to the previous valid count.
        """
        source_views = [image]
        if quickbar_mode == "black_on_white":
            # Coordinates are relative to the 34x15 HP source region above.
            # Scale them to the actual crop so this also works when the game
            # window is captured at a non-1.0 UI scale.
            height, width = image.shape[:2]

            def hp_view(x1, y1, x2, y2):
                left = max(0, min(width, round(x1 * width / 34)))
                right = max(0, min(width, round(x2 * width / 34)))
                top = max(0, min(height, round(y1 * height / 15)))
                bottom = max(0, min(height, round(y2 * height / 15)))
                if right <= left or bottom <= top:
                    return None
                return image[top:bottom, left:right]

            source_views = [
                hp_view(1, 1, 31, 11),
                hp_view(1, 1, 33, 13),
                hp_view(1, 0, 31, 12),
                hp_view(1, 2, 34, 15),
            ]
            source_views = [view for view in source_views if view is not None]

        variants = []

        for crop_view in source_views:
            gray = cv2.cvtColor(crop_view, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for source in (crop_view, gray_bgr):
                variants.append(cv2.resize(
                    source,
                    None,
                    fx=2,
                    fy=2,
                    interpolation=cv2.INTER_NEAREST,
                ))
                for scale in (3, 4):
                    enlarged = cv2.resize(
                        source,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                    variants.append(cv2.copyMakeBorder(
                        enlarged,
                        4,
                        4,
                        4,
                        4,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    ))

        candidates = []
        raw_texts = []
        for variant in variants:
            text, confidence = self._model.recognize(variant)
            raw_texts.append(text)
            digits = re.sub(r'\D', '', text)
            if not digits or len(digits) > 5:
                continue
            candidates.append((int(digits), float(confidence), digits, text))

        if not candidates:
            return None, 0.0, " | ".join(raw_texts)

        if previous_value is None:
            value_counts = {}
            for value, _, _, _ in candidates:
                value_counts[value] = value_counts.get(value, 0) + 1

            supported_values = [
                value for value, count in value_counts.items() if count >= 2
            ]
            if supported_values:
                target_length = max(len(str(value)) for value in supported_values)
                supported_values = [
                    value for value in supported_values
                    if len(str(value)) == target_length
                ]
                pool = [item for item in candidates if item[0] in supported_values]
            else:
                target_length = max(len(item[2]) for item in candidates)
                pool = [item for item in candidates if len(item[2]) == target_length]
        else:
            # Compare the numeric distance instead of locking the digit count.
            # This permits valid boundaries such as 1000 -> 999 and 100 -> 99.
            pool = [
                item for item in candidates
                if abs(item[0] - previous_value) <= 4
            ]
            if not pool:
                return None, 0.0, " | ".join(raw_texts)

        grouped = {}
        for item in pool:
            grouped.setdefault(item[0], []).append(item)

        best_frequency = max(len(items) for items in grouped.values())
        winning_values = [
            value for value, items in grouped.items()
            if len(items) == best_frequency
        ]

        if previous_value is not None and len(winning_values) > 1:
            # This method is only called when the crop pixels changed.  If the
            # old and new number receive equal votes, preferring the old value
            # makes a real one-bottle use invisible forever.  Temporal
            # confirmation in EconomyTracker still rejects one-frame noise.
            changed_values = [value for value in winning_values if value != previous_value]
            if changed_values:
                winning_values = changed_values

        def value_score(value):
            items = grouped[value]
            confidence_sum = sum(item[1] for item in items)
            confidence_max = max(item[1] for item in items)
            if previous_value is None:
                distance_score = 0
            else:
                distance_score = -abs(value - previous_value)
            return confidence_sum, confidence_max, distance_score

        value = max(winning_values, key=value_score)
        winning_items = grouped[value]
        _, raw_confidence, _, text = max(winning_items, key=lambda item: item[1])

        # Two independent preprocessing views agreeing is stronger than one
        # model confidence value.  Promote that consensus above the UI's 0.8
        # threshold; a lone view must still pass on its own confidence.
        if len(winning_items) >= 2:
            confidence = max(raw_confidence, min(0.99, 0.75 + 0.08 * len(winning_items)))
        else:
            confidence = raw_confidence
        return value, confidence, text

    def _prepare_quickbar_digits(self, image, mode):
        """Normalize the outlined pixel font used by quick-slot counters."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        if mode == "black_on_white":
            binary = 255 - binary
            background = 255
        else:
            background = 0

        prepared = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        pad_y = max(2, round(4 * self._scale))
        pad_x = max(3, round(6 * self._scale))
        return cv2.copyMakeBorder(
            prepared,
            pad_y,
            pad_y,
            pad_x,
            pad_x,
            cv2.BORDER_CONSTANT,
            value=(background, background, background),
        )

    def get_player_experience(self) -> tuple:
        # No UI match / bad geometry -> nothing to read yet.
        if not self.is_available():
            return 0, 0

        img = self._safe_crop(self._exp_box)
        if img is None:
            return 0, 0

        # Skip check
        if np.array_equal(self._exp_last_array, img):
            return self._exp_last, 1

        # Inference
        try:
            text, confidence = self._model.recognize(img)
        except Exception:
            if self._should_log("exp_recognition"):
                logger.exception("Experience recognition failed crop_shape=%s", img.shape)
            return 0, 0

        # Extract the first number-like value containing digits, spaces, and dots
        match = re.search(r'^\D*([\d\s.]*?\d)(?=\D|$)', text)
        if (confidence is None or confidence < 0.8 or match is None) and self._should_log(
                "exp_low_confidence"
        ):
            logger.warning(
                "Low-confidence experience OCR confidence=%s raw_text=%r crop_shape=%s",
                confidence,
                text,
                img.shape,
            )

        if match:
            # Remove non-digit
            digits = re.sub(r'\D', '', match.group(1))
            if digits:
                try:
                    exp_int = int(digits)
                except ValueError:
                    return 0, 0

                self._exp_last = exp_int
                self._exp_last_array = img.copy()
                return exp_int, confidence

        return 0, 0

    def _compute_box(self, anchor, offset) -> tuple:
        x, y = anchor
        x1, y1, x2, y2 = offset

        x1 = round(x + x1 * self._scale)
        y1 = round(y + y1 * self._scale)
        x2 = round(x + x2 * self._scale)
        y2 = round(y + y2 * self._scale)

        return x1, y1, x2, y2

    def update(self, screenshot: np.ndarray) -> None:
        if screenshot is None or screenshot.size == 0 or screenshot.ndim < 2:
            # Nothing usable in this frame
            if self._should_log("invalid_frame"):
                logger.warning(
                    "Received unusable capture frame is_none=%s shape=%s",
                    screenshot is None,
                    getattr(screenshot, "shape", None),
                )
            self._screenshot = screenshot
            self._scale = 0
            self._inventory_available = False
            self._quickbar_available = False
            return

        self._screenshot = screenshot
        size = (screenshot.shape[0], screenshot.shape[1])

        size_changed = self._size != size
        self._size = size

        if not size_changed and self._scale > 0:
            self._update_auxiliary_matches()
            return

        try:
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)

            # Try to detect the UI scale by using template matching -------------------------------
            scales = np.linspace(0.5, 2.0, 30)

            best_score_a = 0
            best_score_b = 0
            best_pos_a = (0, 0)
            best_pos_b = (0, 0)

            for scale in scales:
                resized_a = cv2.resize(self._template_a, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
                resized_b = cv2.resize(self._template_b, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

                result_a = cv2.matchTemplate(gray, resized_a, cv2.TM_CCOEFF_NORMED)
                result_b = cv2.matchTemplate(gray, resized_b, cv2.TM_CCOEFF_NORMED)

                _, score_a, _, pos_a = cv2.minMaxLoc(result_a)
                _, score_b, _, pos_b = cv2.minMaxLoc(result_b)

                if score_a > best_score_a:
                    best_score_a = score_a
                    best_pos_a = pos_a

                if score_b > best_score_b:
                    best_score_b = score_b
                    best_pos_b = pos_b

            if best_score_a < 0.8 or best_score_b < 0.8:  # No match found
                if self._should_log("base_template_not_found"):
                    logger.warning(
                        "Base UI templates not found frame_size=%s score_lv=%.3f score_shop=%.3f",
                        size,
                        best_score_a,
                        best_score_b,
                    )
                self._scale = 0
                self._inventory_available = False
                self._quickbar_available = False
                return

            new_scale = (best_pos_b[0] - best_pos_a[0]) / TEMPLATE_DIST

            if new_scale < 0.1 or new_scale > 4:  # No match found / nonsensical scale
                if self._should_log("invalid_ui_scale"):
                    logger.warning(
                        "Detected nonsensical UI scale=%.3f frame_size=%s positions=%s/%s",
                        new_scale,
                        size,
                        best_pos_a,
                        best_pos_b,
                    )
                self._scale = 0
                self._inventory_available = False
                self._quickbar_available = False
                return

            self._scale = new_scale
            logger.info(
                "Base UI detected frame_size=%s scale=%.3f score_lv=%.3f score_shop=%.3f",
                size,
                self._scale,
                best_score_a,
                best_score_b,
            )
            self._lv_box = self._compute_box(best_pos_a, LV_OFFSET)
            self._exp_box = self._compute_box(best_pos_a, EXP_OFFSET)
            self._update_auxiliary_matches(force=True, gray=gray)

        except Exception:
            # Any failure in template matching (odd frame format, OpenCV error, etc.)
            # should leave the extractor in a well-defined "not detected" state
            # rather than propagating and crashing the capture thread.
            if self._should_log("ui_scale_exception"):
                logger.exception("UI scale detection failed frame_size=%s", size)
            self._scale = 0
            self._inventory_available = False
            self._quickbar_available = False

    def _update_auxiliary_matches(self, force=False, gray=None) -> None:
        self._aux_frame_counter += 1
        if not force and self._aux_frame_counter % AUX_DETECT_INTERVAL_FRAMES:
            return

        if self._scale <= 0 or self._screenshot is None or self._screenshot.size == 0:
            self._inventory_available = False
            self._quickbar_available = False
            return

        if gray is None:
            try:
                gray = cv2.cvtColor(self._screenshot, cv2.COLOR_BGRA2GRAY)
            except Exception:
                if self._should_log("grayscale_conversion"):
                    logger.exception(
                        "Failed to convert capture frame to grayscale shape=%s",
                        getattr(self._screenshot, "shape", None),
                    )
                self._inventory_available = False
                self._quickbar_available = False
                return

        # Inventory can be dragged anywhere, so search the entire frame. The
        # quick bar is fixed in the lower-right area of the game window.
        inventory_match = self._match_template_at_current_scale(gray, self._inventory_template)
        quickbar_match = self._match_template_at_current_scale(
            gray,
            self._quickbar_template,
            search_region=(0.5, 0.5, 1.0, 1.0),
        )

        self._inventory_available = inventory_match[0] >= AUX_TEMPLATE_THRESHOLD
        self._quickbar_available = quickbar_match[0] >= AUX_TEMPLATE_THRESHOLD

        aux_state = (self._inventory_available, self._quickbar_available)
        if aux_state != self._last_aux_state:
            logger.info(
                "Auxiliary UI detection inventory=%s score=%.3f quickbar=%s score=%.3f scale=%.3f",
                self._inventory_available,
                inventory_match[0],
                self._quickbar_available,
                quickbar_match[0],
                self._scale,
            )
            self._last_aux_state = aux_state

        if self._inventory_available:
            inventory_origin = self._widget_origin(
                inventory_match[1], INVENTORY_TITLE_TEMPLATE_ORIGIN
            )
            self._meso_box = self._compute_box(inventory_origin, INVENTORY_MESO_OFFSET)

        if self._quickbar_available:
            quickbar_origin = self._widget_origin(
                quickbar_match[1], QUICKBAR_BOTTOM_TEMPLATE_ORIGIN
            )
            self._hp_potion_box = self._compute_box(quickbar_origin, QUICKBAR_HP_OFFSET)
            self._mp_potion_box = self._compute_box(quickbar_origin, QUICKBAR_MP_OFFSET)

    def _match_template_at_current_scale(self, gray, template, search_region=None) -> tuple:
        resized = cv2.resize(
            template,
            None,
            fx=self._scale,
            fy=self._scale,
            interpolation=cv2.INTER_LANCZOS4,
        )
        offset_x = 0
        offset_y = 0
        search_image = gray
        if search_region is not None:
            x1, y1, x2, y2 = search_region
            offset_x = int(gray.shape[1] * x1)
            offset_y = int(gray.shape[0] * y1)
            end_x = int(gray.shape[1] * x2)
            end_y = int(gray.shape[0] * y2)
            search_image = gray[offset_y:end_y, offset_x:end_x]

        if (
                resized.shape[0] > search_image.shape[0]
                or resized.shape[1] > search_image.shape[1]
        ):
            return 0.0, (0, 0)
        result = cv2.matchTemplate(search_image, resized, cv2.TM_CCOEFF_NORMED)
        _, score, _, position = cv2.minMaxLoc(result)
        return float(score), (position[0] + offset_x, position[1] + offset_y)

    def _widget_origin(self, template_position, template_origin) -> tuple:
        return (
            template_position[0] - template_origin[0] * self._scale,
            template_position[1] - template_origin[1] * self._scale,
        )

    @staticmethod
    def _load_template(path):
        resource_path = get_resource_path(path)
        template = cv2.imread(resource_path, cv2.IMREAD_GRAYSCALE)
        assert template is not None, f"Failed to load template at {resource_path}."

        return template.copy()


class PPOCRv6TinyTextRecognition:
    def __init__(self, model_path, config_path):
        # Load model ------------------------------------------------------------------------------
        self.session = ort.InferenceSession(
            model_path.as_posix(),
            providers=["CPUExecutionProvider"],
        )

        # PP-OCRv6 input: [N, 3, 48, 320] ---------------------------------------------------------
        input_info = self.session.get_inputs()[0]

        self.input_name = input_info.name
        self.input_shape = input_info.shape

        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        if not isinstance(self.input_height, int):
            self.input_height = 48

        if not isinstance(self.input_width, int):
            self.input_width = 320

        # Load PaddleOCR model configuration ------------------------------------------------------
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Load character dictionary ---------------------------------------------------------------
        postprocess = self.config.get("PostProcess", {})
        self.character_dict = postprocess.get("character_dict", [])

        if not self.character_dict:
            raise RuntimeError("No character_dict found in inference.yml")

        # PaddleOCR's CTC decoder uses index 0 as blank.
        self.characters = ["blank"] + self.character_dict

        # PP-OCRv6 uses CTCLabelDecode.
        self.use_space_char = postprocess.get("use_space_char", False)

        if self.use_space_char:
            self.characters.append(" ")

    def preprocess(self, image):
        """
        Convert OpenCV BGR image into PP-OCRv6 input.

        Input:
            BGR uint8 image, H x W x 3

        Output:
            float32 array, 1 x 3 x 48 x 320
        """

        if image is None:
            raise ValueError("Input image is None")

        # Make sure image is 3-channel BGR --------------------------------------------------------
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel image, got {image.shape}")

        h, w = image.shape[:2]
        ratio = w / float(h)
        resized_width = min(int(np.ceil(self.input_height * ratio)), self.input_width)

        # C++ accelerated operations: Resize, normalization ([-1, 1] range), and HWC->CHW transpose
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 127.5,
            size=(resized_width, self.input_height),
            mean=(127.5, 127.5, 127.5),
            swapRB=False,
            crop=False
        )

        # Pad horizontally if the image is smaller than input_width
        if resized_width < self.input_width:
            pad_w = self.input_width - resized_width
            blob = np.pad(blob, ((0, 0), (0, 0), (0, 0), (0, pad_w)), mode='constant', constant_values=0)

        return blob

    def decode(self, prediction):
        # Pred shape: [1, sequence_length, num_classes]

        # Remove batch dimension
        prediction = prediction[0]

        # Get best class at every timestep
        indices = np.argmax(prediction, axis=1)

        # Confidence of selected class
        scores = np.max(prediction, axis=1)

        text = []
        confidence = []

        last_index = 0

        for index, score in zip(indices, scores):

            index = int(index)

            # CTC blank
            if index == 0:
                last_index = index
                continue

            # CTC repeated character
            if index == last_index:
                continue

            if index >= len(self.characters):
                last_index = index
                continue

            text.append(self.characters[index])

            confidence.append(float(score))

            last_index = index

        if confidence:
            avg_confidence = float(np.mean(confidence))
        else:
            avg_confidence = 0.0

        return "".join(text), avg_confidence

    def recognize(self, image):
        input_tensor = self.preprocess(image)

        outputs = self.session.run(None, {self.input_name: input_tensor})
        pred = outputs[0]

        text, confidence = self.decode(pred)

        return text, confidence
