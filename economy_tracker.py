from dataclasses import dataclass

from diagnostics import get_logger


logger = get_logger("economy")


@dataclass(frozen=True)
class EconomySnapshot:
    meso_gained: int
    hp_consumed: int
    mp_consumed: int
    hp_cost: int
    mp_cost: int
    potion_cost: int
    net_profit: int


class EconomyTracker:
    """Tracks meso continuously and potions from two confirmed snapshots.

    Potion OCR is intentionally inactive between the start and end snapshots.
    Intermediate misreads therefore cannot alter the displayed consumption.
    """

    def __init__(self, confirmation_reads: int = 2, max_potion_drop: int = 5):
        self.confirmation_reads = max(1, int(confirmation_reads))
        # Retained for source compatibility with earlier versions. Snapshot
        # mode does not reject the final count based on its distance from the
        # start count; a long training session can legitimately use many items.
        self.max_potion_drop = max(1, int(max_potion_drop))
        self.reset()

    def reset(self) -> None:
        self.initial_meso = None
        self.initial_hp_count = None
        self.initial_mp_count = None
        self.final_hp_count = None
        self.final_mp_count = None
        self.last_meso = None
        self.last_hp_count = None
        self.last_mp_count = None
        self.meso_gained = 0
        self.hp_consumed = 0
        self.mp_consumed = 0
        self.potion_phase = "start"
        self._pending = {}

    def update(self, meso=None, hp_count=None, mp_count=None) -> bool:
        changed = False

        confirmed_meso = self._confirm("meso", meso)
        if confirmed_meso is not None:
            if self.initial_meso is None:
                self.initial_meso = confirmed_meso

            net_meso_change = confirmed_meso - self.initial_meso
            if (
                self.last_meso is None
                or confirmed_meso != self.last_meso
                or net_meso_change != self.meso_gained
            ):
                changed = True
            self.last_meso = confirmed_meso
            self.meso_gained = net_meso_change

        changed |= self._capture_potion_snapshot("hp", hp_count)
        changed |= self._capture_potion_snapshot("mp", mp_count)
        return changed

    @property
    def has_potion_start(self) -> bool:
        return self.initial_hp_count is not None and self.initial_mp_count is not None

    @property
    def has_potion_end(self) -> bool:
        return self.final_hp_count is not None and self.final_mp_count is not None

    def begin_potion_settlement(self) -> bool:
        """Arm OCR for a second snapshot after training has ended."""
        if not self.has_potion_start:
            return False

        self.final_hp_count = None
        self.final_mp_count = None
        self.potion_phase = "end"
        self._pending.pop("end_hp", None)
        self._pending.pop("end_mp", None)
        logger.info(
            "Potion end snapshot requested start_hp=%s start_mp=%s",
            self.initial_hp_count,
            self.initial_mp_count,
        )
        return True

    def settle_potions(self, hp_count, mp_count, source: str = "manual") -> bool:
        """Set the final counts directly, normally from the manual dialog."""
        if not self.has_potion_start:
            return False

        try:
            hp = int(hp_count)
            mp = int(mp_count)
        except (TypeError, ValueError):
            return False
        if hp < 0 or mp < 0:
            return False

        self.final_hp_count = hp
        self.final_mp_count = mp
        self.last_hp_count = hp
        self.last_mp_count = mp
        self.hp_consumed = self.initial_hp_count - hp
        self.mp_consumed = self.initial_mp_count - mp
        self.potion_phase = "settled"
        self._pending.pop("end_hp", None)
        self._pending.pop("end_mp", None)
        logger.info(
            "Potion snapshot settled source=%s start_hp=%s end_hp=%s start_mp=%s end_mp=%s hp_consumed=%s mp_consumed=%s",
            source,
            self.initial_hp_count,
            hp,
            self.initial_mp_count,
            mp,
            self.hp_consumed,
            self.mp_consumed,
        )
        return True

    def snapshot(self, hp_price: int = 0, mp_price: int = 0) -> EconomySnapshot:
        hp_cost = self.hp_consumed * max(0, int(hp_price))
        mp_cost = self.mp_consumed * max(0, int(mp_price))
        potion_cost = hp_cost + mp_cost
        return EconomySnapshot(
            meso_gained=self.meso_gained,
            hp_consumed=self.hp_consumed,
            mp_consumed=self.mp_consumed,
            hp_cost=hp_cost,
            mp_cost=mp_cost,
            potion_cost=potion_cost,
            net_profit=self.meso_gained - potion_cost,
        )

    def _capture_potion_snapshot(self, kind: str, value) -> bool:
        phase = self.potion_phase
        if phase not in ("start", "end"):
            return False

        target_attr = (
            f"initial_{kind}_count" if phase == "start"
            else f"final_{kind}_count"
        )
        if getattr(self, target_attr) is not None:
            return False

        confirmed = self._confirm(f"{phase}_{kind}", value)
        if confirmed is None:
            return False

        setattr(self, target_attr, confirmed)
        setattr(self, f"last_{kind}_count", confirmed)
        logger.info(
            "Potion snapshot captured phase=%s kind=%s count=%s",
            phase,
            kind,
            confirmed,
        )

        if phase == "start" and self.has_potion_start:
            self.potion_phase = "ready"
            logger.info(
                "Potion start snapshot ready hp=%s mp=%s",
                self.initial_hp_count,
                self.initial_mp_count,
            )
        elif phase == "end" and self.has_potion_end:
            self.settle_potions(
                self.final_hp_count,
                self.final_mp_count,
                source="ocr",
            )
        return True

    def _confirm(self, key: str, value):
        if value is None:
            self._pending.pop(key, None)
            return None

        try:
            normalized = int(value)
        except (TypeError, ValueError):
            self._pending.pop(key, None)
            return None

        if normalized < 0:
            self._pending.pop(key, None)
            return None

        candidate, reads = self._pending.get(key, (None, 0))
        if candidate == normalized:
            reads += 1
        else:
            candidate, reads = normalized, 1

        if reads >= self.confirmation_reads:
            self._pending.pop(key, None)
            return normalized

        self._pending[key] = (candidate, reads)
        return None

