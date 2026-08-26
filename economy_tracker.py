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

    def __init__(self, confirmation_reads: int = 2, max_potion_drop: int = 5):
        self.confirmation_reads = max(1, int(confirmation_reads))
        # Keep the public argument/property name for compatibility with 1.1.0,
        # but apply the limit symmetrically to both gains and drops.
        self.max_potion_drop = max(1, int(max_potion_drop))
        self.reset()

    def reset(self) -> None:
        self.initial_meso = None
        self.initial_hp_count = None
        self.initial_mp_count = None
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
        initial_attr = f"initial_{kind}_count"
        previous = getattr(self, attr)

        if previous is None:
            setattr(self, initial_attr, confirmed)
            setattr(self, attr, confirmed)
            return False

        change = confirmed - previous
        if abs(change) >= self.max_potion_drop:
            # A two-frame OCR error can still pass the ordinary confirmation
            # filter. Do not rebase on it: recovery from a false baseline can
            # otherwise look like a pickup (for example 1500 -> 1508 = -8).
            # Press Reset after intentionally replacing a large stack.
            return False

        if confirmed == previous:
            return False

        initial = getattr(self, initial_attr)
        consumed_attr = f"{kind}_consumed"
        new_consumed = initial - confirmed
        changed = new_consumed != getattr(self, consumed_attr)
        setattr(self, attr, confirmed)
        setattr(self, consumed_attr, new_consumed)
        return changed

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

