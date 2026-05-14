import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from model import World, Infantry, Archer, Cavalry, Formation


class WorldView:
    TERRAIN_COLORS = {
        0: (0, 200, 0),      # Plains - green
        1: (0, 100, 255),    # Water - blue
        2: (139, 69, 19),    # Forest - brown
    }

    UNIT_COLORS = {
        'Infantry': '#ff0000',
        'Archer':   '#ffff00',
        'Cavalry':  '#ff8800',
    }

    # ── Metrics panel palette ─────────────────────────────────────
    BG            = '#1a1a2e'
    BG_SECTION    = '#16213e'
    TEXT_PRIMARY  = '#e0e0e0'
    TEXT_DIM      = '#888899'
    FRIENDLY_CLR  = '#ff4444'
    ENEMY_CLR     = '#4488ff'
    BAR_BG        = '#2a2a3e'
    HP_FRIENDLY   = '#cc3333'
    HP_ENEMY      = '#3366cc'
    COHESION_CLR  = '#44bb88'
    WALL_ON       = '#ffcc00'
    WALL_OFF      = '#444455'
    KILL_FRIENDLY = '#ff6666'
    KILL_ENEMY    = '#6699ff'

    PANEL_W = 280

    def __init__(self, root: tk.Tk, world: World, tile_size: int = 10):
        self.world = world
        self.tile_size = tile_size

        # ── Track initial state for kill/hp percentage calcs ──────
        self._initial_friendly_hp = 0
        self._initial_enemy_hp = 0
        self._initial_friendly_count = 0
        self._initial_enemy_count = 0
        self._initial_formation_counts: dict[int, int] = {}
        self._snapshot_initials()

        # Kill tracking
        self._friendly_kills = 0
        self._enemy_kills = 0
        self._prev_friendly_alive = self._initial_friendly_count
        self._prev_enemy_alive = self._initial_enemy_count
        self._tick_number = 0

        # ── Layout: battle canvas left, metrics right ─────────────
        canvas_width = world.width * tile_size
        canvas_height = world.height * tile_size

        self.container = tk.Frame(root, bg=self.BG)
        self.container.pack(fill='both', expand=True)

        self.canvas = tk.Canvas(
            self.container, width=canvas_width, height=canvas_height,
            bg='#000000', highlightthickness=0,
        )
        self.canvas.pack(side='left')

        self.panel_height = canvas_height
        self.panel = tk.Canvas(
            self.container, width=self.PANEL_W, height=self.panel_height,
            bg=self.BG, highlightthickness=0,
        )
        self.panel.pack(side='left', fill='y')

        self._build_terrain_image()

    # ── Initial snapshot ──────────────────────────────────────────

    def _snapshot_initials(self):
        for team in self.world.get_teams():
            for formation in team.get_formations():
                fid = id(formation)
                count = len(formation.get_units())
                self._initial_formation_counts[fid] = count

                if not team.team_id:
                    self._initial_friendly_hp += sum(
                        u.get_hp() for u in formation.get_units()
                    )
                    self._initial_friendly_count += count
                else:
                    self._initial_enemy_hp += sum(
                        u.get_hp() for u in formation.get_units()
                    )
                    self._initial_enemy_count += count

    def reset_tracking(self):
        """Call after regenerate to re-snapshot."""
        self._initial_friendly_hp = 0
        self._initial_enemy_hp = 0
        self._initial_friendly_count = 0
        self._initial_enemy_count = 0
        self._initial_formation_counts.clear()
        self._snapshot_initials()

        self._friendly_kills = 0
        self._enemy_kills = 0
        self._prev_friendly_alive = self._initial_friendly_count
        self._prev_enemy_alive = self._initial_enemy_count
        self._tick_number = 0

    # ── Terrain ───────────────────────────────────────────────────

    def _build_terrain_image(self):
        img = Image.new('RGB', (self.world.width, self.world.height))
        pixels = img.load()
        for y in range(self.world.height):
            for x in range(self.world.width):
                terrain = int(self.world.tiles[y, x, 0])
                height  = int(self.world.tiles[y, x, 1])
                base = self.TERRAIN_COLORS[terrain]
                brightness = 0.4 + (height / 5) * 0.6
                pixels[x, y] = tuple(int(c * brightness) for c in base)
        img = img.resize(
            (self.world.width * self.tile_size,
             self.world.height * self.tile_size),
            Image.NEAREST,
        )
        self._terrain_photo = ImageTk.PhotoImage(img)

    def rebuild_terrain(self):
        self._build_terrain_image()
        self.reset_tracking()

    # ── Main render ───────────────────────────────────────────────

    def render(self):
        self._render_battle()
        self._update_kill_tracking()
        self._render_panel()
        self._tick_number += 1

    # ── Formation destination zone colours ───────────────────────
    DEST_FRIENDLY = '#ff4444'
    DEST_ENEMY    = '#4488ff'
    DEST_MARKER   = 4          # endpoint marker radius in pixels

    def _render_battle(self):
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self._terrain_photo)

        ts = self.tile_size

        # ── Destination zones (drawn first so units sit on top) ───
        self._draw_destination_zones(ts)

        # ── Units ─────────────────────────────────────────────────
        half = ts // 2
        for unit in self.world.get_all_units():
            if unit.is_dead():
                continue
            unit_type = type(unit).__name__
            color = self.UNIT_COLORS.get(unit_type, '#ffffff')
            outline = '#0000ff' if unit.get_team() else '#ff0000'

            cx = unit.get_x() * ts
            cy = unit.get_y() * ts

            if unit_type == 'Infantry':
                self.canvas.create_rectangle(
                    cx - half, cy - half, cx + half, cy + half,
                    fill=color, outline=outline, width=2,
                )
            elif unit_type == 'Archer':
                self.canvas.create_oval(
                    cx - half, cy - half, cx + half, cy + half,
                    fill=color, outline=outline, width=2,
                )
            elif unit_type == 'Cavalry':
                self.canvas.create_polygon(
                    cx, cy - half,
                    cx - half, cy + half,
                    cx + half, cy + half,
                    fill=color, outline=outline, width=2,
                )
            else:
                self.canvas.create_oval(
                    cx - half, cy - half, cx + half, cy + half,
                    fill=color, outline=outline, width=2,
                )

    def _draw_destination_zones(self, ts: int):
        """
        For each living formation, draw a stippled rectangle showing
        where the formation is moving to.

        The rectangle spans from the port–starboard front line back
        by (num_ranks * rank_spacing) in the depth (normal) direction.
        Port and starboard endpoints get small square markers.
        """
        import math

        for team in self.world.get_teams():
            zone_color = self.DEST_ENEMY if team.team_id else self.DEST_FRIENDLY

            for formation in team.get_formations():
                living = formation.get_living_units()
                if not living:
                    continue

                px, py = formation.get_port()
                sx, sy = formation.get_starboard()

                dx = sx - px
                dy = sy - py
                length = math.hypot(dx, dy)
                if length < 1e-9:
                    continue

                # Tangent and normal (depth direction)
                tx = dx / length
                ty = dy / length
                nx = ty
                ny = -tx

                # Rank spacing: infantry uses 1.0, others 2.0
                # (mirrors the adjustment in Formation.move)
                if isinstance(living[0], Infantry):
                    rank_spacing = 1.0
                else:
                    rank_spacing = 2.0

                num_ranks = formation.get_rank_count()
                depth = max(num_ranks, 1) * rank_spacing

                # Four corners of the formation rectangle in world coords
                # Front: port → starboard
                # Back: port+depth → starboard+depth
                corners_world = [
                    (px,             py),
                    (sx,             sy),
                    (sx + depth * nx, sy + depth * ny),
                    (px + depth * nx, py + depth * ny),
                ]

                # Convert to canvas pixels and flatten for create_polygon
                coords = []
                for wx, wy in corners_world:
                    coords.append(wx * ts)
                    coords.append(wy * ts)

                # Stippled fill for translucency
                self.canvas.create_polygon(
                    *coords,
                    fill=zone_color, outline=zone_color,
                    stipple='gray25', width=1,
                )

                # Front line (solid, slightly thicker)
                self.canvas.create_line(
                    px * ts, py * ts, sx * ts, sy * ts,
                    fill=zone_color, width=2,
                )

                # Port and starboard endpoint markers
                r = self.DEST_MARKER
                # Port marker (square)
                cpx, cpy = px * ts, py * ts
                self.canvas.create_rectangle(
                    cpx - r, cpy - r, cpx + r, cpy + r,
                    fill=zone_color, outline='#ffffff', width=1,
                )
                # Starboard marker (circle to distinguish)
                csx, csy = sx * ts, sy * ts
                self.canvas.create_oval(
                    csx - r, csy - r, csx + r, csy + r,
                    fill=zone_color, outline='#ffffff', width=1,
                )

    # ── Kill tracking ─────────────────────────────────────────────

    def _update_kill_tracking(self):
        friendly_alive = 0
        enemy_alive = 0
        for team in self.world.get_teams():
            alive = len(team.get_living_units())
            if not team.team_id:
                friendly_alive = alive
            else:
                enemy_alive = alive

        # Deaths on enemy side = friendly kills
        enemy_died = self._prev_enemy_alive - enemy_alive
        if enemy_died > 0:
            self._friendly_kills += enemy_died

        friendly_died = self._prev_friendly_alive - friendly_alive
        if friendly_died > 0:
            self._enemy_kills += friendly_died

        self._prev_friendly_alive = friendly_alive
        self._prev_enemy_alive = enemy_alive

    # ── Metrics panel ─────────────────────────────────────────────

    def _render_panel(self):
        p = self.panel
        p.delete('all')

        W = self.PANEL_W
        margin = 12
        bar_h = 14
        usable = W - 2 * margin
        y = margin

        # ── Helper closures ───────────────────────────────────────

        def heading(text, yy):
            p.create_text(margin, yy, text=text, anchor='nw',
                          fill=self.TEXT_PRIMARY,
                          font=('Consolas', 11, 'bold'))
            return yy + 20

        def label_value(label, value, yy, color=None):
            color = color or self.TEXT_DIM
            p.create_text(margin, yy, text=label, anchor='nw',
                          fill=self.TEXT_DIM, font=('Consolas', 9))
            p.create_text(W - margin, yy, text=str(value), anchor='ne',
                          fill=color, font=('Consolas', 9, 'bold'))
            return yy + 16

        def bar(fraction, yy, color, h=bar_h):
            fraction = max(0.0, min(1.0, fraction))
            # Background
            p.create_rectangle(margin, yy, margin + usable, yy + h,
                               fill=self.BAR_BG, outline='')
            # Fill
            if fraction > 0:
                p.create_rectangle(margin, yy, margin + usable * fraction,
                                   yy + h, fill=color, outline='')
            return yy + h + 4

        def dual_bar(frac_left, frac_right, yy, clr_left, clr_right,
                     h=bar_h):
            """Two bars facing each other from centre."""
            frac_left = max(0.0, min(1.0, frac_left))
            frac_right = max(0.0, min(1.0, frac_right))
            mid = margin + usable // 2
            half = usable // 2

            p.create_rectangle(margin, yy, margin + usable, yy + h,
                               fill=self.BAR_BG, outline='')
            # Left fills leftward from centre
            if frac_left > 0:
                p.create_rectangle(
                    mid - half * frac_left, yy, mid, yy + h,
                    fill=clr_left, outline='',
                )
            # Right fills rightward from centre
            if frac_right > 0:
                p.create_rectangle(
                    mid, yy, mid + half * frac_right, yy + h,
                    fill=clr_right, outline='',
                )
            # Centre line
            p.create_line(mid, yy, mid, yy + h, fill='#ffffff', width=1)
            return yy + h + 4

        def separator(yy):
            p.create_line(margin, yy, W - margin, yy,
                          fill='#333344', width=1)
            return yy + 8

        # ── Gather data ───────────────────────────────────────────

        friendly_team = None
        enemy_team = None
        for team in self.world.get_teams():
            if not team.team_id:
                friendly_team = team
            else:
                enemy_team = team

        if not friendly_team or not enemy_team:
            return

        f_hp = sum(u.get_hp() for u in friendly_team.get_living_units())
        e_hp = sum(u.get_hp() for u in enemy_team.get_living_units())
        f_alive = len(friendly_team.get_living_units())
        e_alive = len(enemy_team.get_living_units())

        f_hp_frac = f_hp / max(self._initial_friendly_hp, 1)
        e_hp_frac = e_hp / max(self._initial_enemy_hp, 1)
        f_alive_frac = f_alive / max(self._initial_friendly_count, 1)
        e_alive_frac = e_alive / max(self._initial_enemy_count, 1)

        # ── SECTION: Overview ─────────────────────────────────────
        y = heading('BATTLE OVERVIEW', y)

        y = label_value('Tick', self._tick_number, y)
        y += 2

        # HP comparison bar
        p.create_text(margin, y, text='HP', anchor='nw',
                      fill=self.TEXT_DIM, font=('Consolas', 9))
        p.create_text(W - margin, y, text=f'{f_hp} vs {e_hp}', anchor='ne',
                      fill=self.TEXT_DIM, font=('Consolas', 8))
        y += 14
        y = dual_bar(f_hp_frac, e_hp_frac, y,
                     self.HP_FRIENDLY, self.HP_ENEMY)

        # Alive comparison bar
        p.create_text(margin, y, text='Alive', anchor='nw',
                      fill=self.TEXT_DIM, font=('Consolas', 9))
        p.create_text(
            W - margin, y,
            text=f'{f_alive}/{self._initial_friendly_count}'
                 f' vs {e_alive}/{self._initial_enemy_count}',
            anchor='ne', fill=self.TEXT_DIM, font=('Consolas', 8),
        )
        y += 14
        y = dual_bar(f_alive_frac, e_alive_frac, y,
                     self.HP_FRIENDLY, self.HP_ENEMY)

        y = separator(y)

        # ── SECTION: Kills ────────────────────────────────────────
        y = heading('KILLS', y)

        ticks = max(self._tick_number, 1)
        f_kpm = self._friendly_kills / ticks * 60
        e_kpm = self._enemy_kills / ticks * 60

        y = label_value('Friendly kills', self._friendly_kills, y,
                        self.KILL_FRIENDLY)
        y = label_value('Enemy kills', self._enemy_kills, y,
                        self.KILL_ENEMY)

        # Kill rate bars (normalise to whichever is higher)
        max_kills = max(self._friendly_kills, self._enemy_kills, 1)
        p.create_text(margin, y, text='Kill ratio', anchor='nw',
                      fill=self.TEXT_DIM, font=('Consolas', 9))
        y += 14
        y = dual_bar(
            self._friendly_kills / max_kills,
            self._enemy_kills / max_kills,
            y, self.KILL_FRIENDLY, self.KILL_ENEMY,
        )

        y = label_value('F kill/tick', f'{f_kpm:.1f}/min', y,
                        self.TEXT_DIM)
        y = label_value('E kill/tick', f'{e_kpm:.1f}/min', y,
                        self.TEXT_DIM)

        y = separator(y)

        # ── SECTION: Formations ───────────────────────────────────
        y = heading('FRIENDLY FORMATIONS', y)
        y = self._render_formation_list(
            friendly_team.get_formations(), y, self.FRIENDLY_CLR,
        )

        y = separator(y)

        y = heading('ENEMY FORMATIONS', y)
        y = self._render_formation_list(
            enemy_team.get_formations(), y, self.ENEMY_CLR,
        )

    def _render_formation_list(self, formations, y, team_color):
        p = self.panel
        W = self.PANEL_W
        margin = 12
        usable = W - 2 * margin

        for i, formation in enumerate(formations):
            living = formation.get_living_units()
            all_units = formation.get_units()
            total = self._initial_formation_counts.get(
                id(formation), len(all_units),
            )
            alive = len(living)

            if not all_units:
                continue

            # Type name
            unit_type = type(all_units[0]).__name__
            type_color = self.UNIT_COLORS.get(unit_type, '#aaaaaa')

            # ── Header line: "F0: Infantry  12/30" ────────────────
            label = f'F{i}: {unit_type}'
            p.create_text(margin, y, text=label, anchor='nw',
                          fill=type_color, font=('Consolas', 9, 'bold'))
            count_str = f'{alive}/{total}'
            p.create_text(W - margin, y, text=count_str, anchor='ne',
                          fill=self.TEXT_PRIMARY if alive > 0 else '#553333',
                          font=('Consolas', 9))
            y += 15

            # ── HP bar ────────────────────────────────────────────
            if total > 0:
                frac = alive / total
                p.create_rectangle(margin, y, margin + usable, y + 8,
                                   fill=self.BAR_BG, outline='')
                if frac > 0:
                    p.create_rectangle(
                        margin, y, margin + usable * frac, y + 8,
                        fill=team_color, outline='',
                    )
                y += 12

            if not living:
                p.create_text(margin + 4, y, text='DESTROYED',
                              anchor='nw', fill='#553333',
                              font=('Consolas', 8, 'italic'))
                y += 16
                continue

            # ── Position ──────────────────────────────────────────
            xs = [u.get_x() for u in living]
            ys_vals = [u.get_y() for u in living]
            avg_x = sum(xs) / len(xs)
            avg_y = sum(ys_vals) / len(ys_vals)
            p.create_text(margin + 4, y,
                          text=f'pos ({avg_x:.0f}, {avg_y:.0f})',
                          anchor='nw', fill=self.TEXT_DIM,
                          font=('Consolas', 8))
            y += 13

            # ── Cohesion bar ──────────────────────────────────────
            cohesion = formation.measure_formation_cohesion()
            coh_w = usable * 0.6
            p.create_text(margin + 4, y, text='coh', anchor='nw',
                          fill=self.TEXT_DIM, font=('Consolas', 8))
            bx = margin + 30
            p.create_rectangle(bx, y + 1, bx + coh_w, y + 9,
                               fill=self.BAR_BG, outline='')
            if cohesion > 0:
                p.create_rectangle(
                    bx, y + 1, bx + coh_w * cohesion, y + 9,
                    fill=self.COHESION_CLR, outline='',
                )
            p.create_text(bx + coh_w + 4, y,
                          text=f'{cohesion:.0%}', anchor='nw',
                          fill=self.COHESION_CLR, font=('Consolas', 8))
            y += 13

            # ── Shield wall indicator ─────────────────────────────
            if isinstance(living[0], Infantry):
                wall_on = living[0].get_wall()
                dot_color = self.WALL_ON if wall_on else self.WALL_OFF
                label_text = 'WALL ON' if wall_on else 'wall off'
                p.create_oval(margin + 4, y + 2, margin + 10, y + 8,
                              fill=dot_color, outline='')
                p.create_text(margin + 14, y,
                              text=label_text, anchor='nw',
                              fill=dot_color, font=('Consolas', 8))
                y += 13

            y += 4  # spacing between formations

        return y