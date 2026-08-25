from dataclasses import dataclass


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
    """Tracks confirmed wallet and quick-slot counter changes for one session."""

    def __init__(self, confirmation_reads: int = 2, max_potion_drop: int = 10):
        self.confirmation_reads = max(1, int(confirmation_reads))
        self.max_potion_drop = max(1, int(max_potion_drop))
        self.reset()

    def reset(self) -> None:
        self.last_meso = None
        self.last_hp_count = None
        self.last_mp_count = None
        self.meso_gained = 0
        self.hp_consumed = 0
        self.mp_consumed = 0
        self._pending = {}

    def update(self, meso=None, hp_count=None, mp_count=None) -> bool:
        changed = False

        confirmed_meso = self._confirm("meso", meso)
        if confirmed_meso is not None:
            if self.last_meso is None or confirmed_meso != self.last_meso:
                changed = True
            if self.last_meso is not None and confirmed_meso > self.last_meso:
                self.meso_gained += confirmed_meso - self.last_meso
            self.last_meso = confirmed_meso

        changed |= self._update_potion("hp", hp_count)
        changed |= self._update_potion("mp", mp_count)
        return changed

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

    def _update_potion(self, kind: str, value) -> bool:
        confirmed = self._confirm(kind, value)
        if confirmed is None:
            return False

        attr = f"last_{kind}_count"
        previous = getattr(self, attr)

        if previous is None:
            setattr(self, attr, confirmed)
            return False

        if confirmed < previous:
            consumed = previous - confirmed
            # The quick bar is sampled continuously. A large one-step decrease is
            # almost certainly a missing OCR digit, not real potion consumption.
            if consumed > self.max_potion_drop:
                return False
            if kind == "hp":
                self.hp_consumed += consumed
            else:
                self.mp_consumed += consumed
            setattr(self, attr, confirmed)
            return True

        # Counter increases are restocks and establish a new baseline.
        if confirmed > previous:
            setattr(self, attr, confirmed)
        return False

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
