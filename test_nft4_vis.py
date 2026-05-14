"""
NFT-4: Visualiser Frame Rate Under Load
========================================
Non-functional / Performance

Measures the wall-clock time of each WorldView.render() call during a
full battle with a large unit count. Asserts that the average render
time stays below 100 ms (10 fps minimum) and the worst single frame
stays below 300 ms.

Requires a display — cannot run headlessly inside pytest.

Run directly:
    python test_nft4_vis.py

How it works
------------
WorldView.render() is monkey-patched to record the duration of every
call. The Controller's _do_tick() drives one world.tick() + view.render()
per call. We invoke _do_tick() FRAME_COUNT times via root.after() so
tkinter's event loop stays alive and the Canvas actually flushes draws
rather than just queuing them. When all frames are done the window
closes automatically and results are printed.
"""

import sys
import time
import tkinter as tk

# ── Configuration ──────────────────────────────────────────────────────────
FRAME_COUNT    = 200   # number of frames to measure
MAX_AVG_MS     = 100.0 # average render time threshold in ms  (≈ 10 fps)
MAX_PEAK_MS    = 300.0 # single worst-frame threshold in ms
UNITS_PER_SIDE = 200   # large enough to stress the renderer
MS_PER_FRAME   = 0     # 0 = run as fast as possible during the test
# ──────────────────────────────────────────────────────────────────────────


def _make_large_world(units_per_side: int):
    """
    Build a World with large armies without relying on populate_team,
    which is capped by ARMY_PRESETS (max ~50 units per formation).
    One infantry formation per side, placed on opposite edges and given
    a march destination so combat begins during the test.
    """
    import model

    # tile_size=8 → 100-unit world = 800 px wide, manageable window size
    world = model.World(100, 100, tick_rate=10, curriculum_stage=0)

    for team_id in [False, True]:
        team = model.Team(team_id=team_id)
        units = [model.Infantry() for _ in range(units_per_side)]
        for u in units:
            u.set_team(team_id)

        formation = model.Formation(units)

        if not team_id:
            formation.move(5, 5, 5, 95)          # left side
            for u in formation.get_units():
                dest = u.get_destination()
                if dest:
                    u.set_x(dest[0])
                    u.set_y(dest[1])
            for u in formation.get_units():       # march right
                u.set_destination(90, u.get_y())
        else:
            formation.move(95, 5, 95, 95)         # right side
            for u in formation.get_units():
                dest = u.get_destination()
                if dest:
                    u.set_x(dest[0])
                    u.set_y(dest[1])
            for u in formation.get_units():       # march left
                u.set_destination(10, u.get_y())

        team.add_formation(formation)
        world.add_team(team)

    return world


def run_nft4():
    from view import WorldView
    from controller import Controller

    world = _make_large_world(UNITS_PER_SIDE)

    root = tk.Tk()
    root.title(
        f"NFT-4  {UNITS_PER_SIDE}v{UNITS_PER_SIDE} frame-rate test "
        f"— closes automatically after {FRAME_COUNT} frames"
    )
    root.bind('<Escape>', lambda e: root.quit())

    view       = WorldView(root, world, tile_size=8)
    controller = Controller(world, view, root, ms_per_frame=MS_PER_FRAME)

    # ── Patch WorldView.render() to record per-call duration ──────────────
    # We measure the duration of render() itself, not the full tick, so
    # the result reflects rendering cost independently of simulation cost.
    render_durations_ms = []
    _original_render = view.render

    def _timed_render():
        t0 = time.perf_counter()
        _original_render()
        render_durations_ms.append((time.perf_counter() - t0) * 1000)

    view.render = _timed_render

    # ── Drive frames via root.after() ────────────────────────────────────
    # Calling root.after() keeps the tkinter event loop alive between
    # frames, which is necessary for the Canvas to actually flush draw
    # commands to the display. A plain Python loop would queue everything
    # without rendering until after the loop exits.
    frames_done = [0]

    def _step():
        if frames_done[0] >= FRAME_COUNT:
            root.quit()
            return

        over = controller._do_tick()
        frames_done[0] += 1
        root.after(MS_PER_FRAME, _step)

    view.render()                    # show initial state before loop
    root.after(MS_PER_FRAME, _step)

    print(
        f"[NFT-4] Measuring {FRAME_COUNT} frames "
        f"({UNITS_PER_SIDE}v{UNITS_PER_SIDE} infantry) ..."
    )
    root.mainloop()

    # ── Analyse ───────────────────────────────────────────────────────────
    n = len(render_durations_ms)
    if n < 2:
        print(f"[NFT-4] SKIP — only {n} frame(s) recorded.")
        sys.exit(0)

    avg_ms  = sum(render_durations_ms) / n
    peak_ms = max(render_durations_ms)
    min_ms  = min(render_durations_ms)

    sorted_d = sorted(render_durations_ms)
    p95_ms   = sorted_d[int(n * 0.95)]

    print(f"\n[NFT-4] Results — {n} frames, {UNITS_PER_SIDE}v{UNITS_PER_SIDE} units")
    print(f"        Average  : {avg_ms:.2f} ms  (threshold ≤ {MAX_AVG_MS} ms)")
    print(f"        95th pct : {p95_ms:.2f} ms")
    print(f"        Peak     : {peak_ms:.2f} ms  (threshold ≤ {MAX_PEAK_MS} ms)")
    print(f"        Min      : {min_ms:.2f} ms")
    print(f"        ~FPS     : {1000 / avg_ms:.1f} avg")

    passed = avg_ms <= MAX_AVG_MS and peak_ms <= MAX_PEAK_MS

    if passed:
        print("[NFT-4] PASS")
        sys.exit(0)
    else:
        reasons = []
        if avg_ms > MAX_AVG_MS:
            reasons.append(
                f"average {avg_ms:.1f} ms exceeds {MAX_AVG_MS} ms"
            )
        if peak_ms > MAX_PEAK_MS:
            reasons.append(
                f"peak {peak_ms:.1f} ms exceeds {MAX_PEAK_MS} ms"
            )
        print(f"[NFT-4] FAIL — {'; '.join(reasons)}")
        sys.exit(1)


if __name__ == "__main__":
    run_nft4()