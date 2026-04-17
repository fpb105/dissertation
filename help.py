import time
import math
from collections import defaultdict
from model import World, Team

# ── Timing decorator ──────────────────────────────────────────────────

call_times = defaultdict(lambda: {"total": 0.0, "calls": 0})


def timed(cls):
    """Wrap every public method on cls with a timer that accumulates."""
    for name in list(vars(cls)):
        if name.startswith("__"):
            continue

        raw = vars(cls)[name]

        # Skip staticmethods and non-callables
        if isinstance(raw, staticmethod):
            continue
        if not callable(raw):
            continue

        original = getattr(cls, name)

        def make_wrapper(fn, fn_name):
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                call_times[fn_name]["total"] += elapsed
                call_times[fn_name]["calls"] += 1
                return result
            return wrapper

        setattr(cls, name, make_wrapper(original, f"World.{name}"))

    return cls


# Patch World
timed(World)

# ── Run simulation ────────────────────────────────────────────────────

world = World(width=100, height=100, tick_rate=10, curriculum_stage=0)

team_a = Team(team_id=False)
team_b = Team(team_id=True)
world.populate_team(team_a, min_formations=2, max_formations=3,
                    min_units=20, max_units=50)
world.populate_team(team_b, min_formations=2, max_formations=3,
                    min_units=20, max_units=50)

for unit in team_a.get_all_units():
    unit.set_destination(world.width - 10, unit.get_y())
for unit in team_b.get_all_units():
    unit.set_destination(10, unit.get_y())

NUM_TICKS = 1000

total_start = time.perf_counter()
for _ in range(NUM_TICKS):
    world.tick()
total_elapsed = time.perf_counter() - total_start

# ── Print results ─────────────────────────────────────────────────────

print(f"\n{'Method':<45} {'Calls':>8} {'Total (s)':>10} {'Per Call (ms)':>14}")
print("─" * 80)

sorted_methods = sorted(call_times.items(), key=lambda x: x[1]["total"], reverse=True)

for name, data in sorted_methods:
    total = data["total"]
    calls = data["calls"]
    per_call = (total / calls * 1000) if calls > 0 else 0
    print(f"{name:<45} {calls:>8} {total:>10.4f} {per_call:>14.4f}")

print("─" * 80)
print(f"{'Total tick loop':<45} {NUM_TICKS:>8} {total_elapsed:>10.4f} {total_elapsed/NUM_TICKS*1000:>14.4f}")