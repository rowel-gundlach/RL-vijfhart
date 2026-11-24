"""Live views of the Q-matrices in demos 2, 4 and 5."""
import ipywidgets as widgets
import numpy as np

from .applied import BROKEN, MACHINE_ACTIONS, Maintenance, best_action

CELL = "padding:4px 14px;text-align:right;border:1px solid #d9e1ea;color:#1f2d3d;"


def _table(corner, columns, rows, note, decimals=1):
    """rows holds (label, values, best column); a value of None means the action is not possible."""
    head = "".join(f'<th style="{CELL}">{name}</th>' for name in [corner, *columns])
    body = ""
    for label, values, best in rows:
        body += f'<tr><th style="{CELL}">{label}</th>'
        for column, value in enumerate(values):
            style = CELL + ("background:#bfe6cd;font-weight:700;" if column == best else "")
            body += f'<td style="{style}">{"–" if value is None else f"{value:,.{decimals}f}"}</td>'
        body += "</tr>"
    return ('<div style="background:#fff;color:#1f2d3d;padding:10px;line-height:1.3">'
            f'<p>{note}</p><table style="border-collapse:collapse"><tr>{head}</tr>{body}</table></div>')


def _snake_picture(game, side=20):
    """A little board: dark = head, blue = body, red = apple."""
    full = game.size * side
    cells = "".join(f'<rect x="{x * side}" y="{y * side}" width="{side}" height="{side}" fill="#fff" stroke="#e3e8ee"/>'
                    for x in range(game.size) for y in range(game.size))
    middle = [(x * side + side / 2, y * side + side / 2) for x, y in game.snake]
    points = " ".join(f"{x},{y}" for x, y in middle)
    body = (f'<polyline points="{points}" fill="none" stroke="#2f6fde" stroke-width="{side - 6}" '
            'stroke-linecap="round" stroke-linejoin="round"/>')
    hx, hy = middle[0]
    head = f'<path d="M{hx} {hy - 9} L{hx + 9} {hy + 7} L{hx - 9} {hy + 7} Z" fill="#1f2d3d"/>'
    ax, ay = game.apple
    apple = f'<circle cx="{ax * side + side / 2}" cy="{ay * side + side / 2}" r="{side * 0.36}" fill="#d64533"/>'
    return (f'<svg width="{full}" height="{full}" viewBox="0 0 {full} {full}" '
            f'style="display:block;border:2px solid #5d6c7c">{cells}{body}{head}{apple}</svg>')


def show_snake_q(demo):
    """Four recognisable rows of the table, as pictures. Follows the demo while it trains."""
    from .snake import SITUATIONS, TURNS, greedy, situation_game, situation_values

    moves = ["↑ straight", "← left", "→ right"]
    games = []
    for name, body, apple in SITUATIONS:
        game = situation_game(body, apple)
        rule = TURNS.index((greedy(game) - game.heading + 1) % 4 - 1)
        games.append((name, game, _snake_picture(game), moves[rule]))
    table = widgets.HTML()
    cell = CELL + "font-size:17px;vertical-align:middle;"
    moment = widgets.ToggleButtons(description="Show:", style={"button_width": "auto"})

    updating = []

    def paint(change=None):
        if updating:                     # changing the buttons below fires this again: ignore that
            return
        options = [("now", "now")] + [(f"after {g:,} games", g) for g in sorted(demo.snapshots)]
        if tuple(moment.options) != tuple(options):
            chosen = moment.value
            updating.append(True)
            moment.options = options
            moment.value = chosen if chosen in [value for _, value in options] else "now"
            updating.clear()
        if moment.value == "now":
            q, played = demo.agent.q, demo.agent.games
        else:
            q, played = demo.snapshots[moment.value], moment.value
        head = "".join(f'<th style="{cell}">{name}</th>' for name in ["what it sees", *moves, "Greedy does"])
        body = ""
        for name, game, picture, rule in games:
            values = situation_values(q, game)
            body += (f'<tr><td style="{cell}text-align:left;width:190px;line-height:1.3">'
                     f'{picture}<div style="margin-top:4px">{name}</div></td>')
            for move in range(len(moves)):
                best = values is not None and move == values.argmax() and (values == values.max()).sum() == 1
                style = cell + ("background:#bfe6cd;font-weight:700;" if best else "")
                body += f'<td style="{style}">{"?" if values is None else f"{values[move]:.1f}"}</td>'
            body += f'<td style="{cell}">{rule}</td></tr>'
        table.value = ('<div style="background:#fff;color:#1f2d3d;padding:10px;line-height:1.3">'
                       f'<p style="font-size:19px"><b>The snake\'s table after {played:,} games</b></p>'
                       f'<table style="border-collapse:collapse"><tr>{head}</tr>{body}</table>'
                       "<p>Q = expected future reward. Green = best move on average, ? = not seen yet. "
                       "Each row is the average of that picture turned in all four directions.</p></div>")

    demo.observe(paint, "view")
    moment.observe(paint, "value")
    paint()
    return widgets.VBox([moment, table])


def show_maintenance_q(demo):
    """One row per state, one column per action. Follows the demo while it trains."""
    table = widgets.HTML()

    def paint(change=None):
        rows = []
        for state in range(BROKEN + 1):
            allowed = Maintenance.allowed(state)
            values = [demo.q[state, a] if a in allowed else None for a in range(len(MACHINE_ACTIONS))]
            best = best_action(demo.q, state) if np.any(demo.q[state]) else None
            rows.append(("broken" if state == BROKEN else f"wear {state}", values, best))
        table.value = _table("state", MACHINE_ACTIONS, rows,
                             "Q = expected discounted future profit in €. Green = best action, – = not possible.")

    demo.observe(paint, "view")
    paint()
    return table


def show_pricing_q(demo, concerts=200, often=0.01, enough_tries=20):
    """The Q-matrix on the final day of selling: one row per number of tickets left, one column per price.

    Only rows that really happen on the final day are shown, and only prices tried often enough there. The rest
    still holds the hopeful starting guess ("every ticket sells at the top price"), which is no real expectation.
    """
    from .applied import run_season
    days, tickets, groups = demo.settings["days"], demo.settings["tickets"], demo.settings["groups"]
    columns = [f"€{price}" for price in demo.settings["prices"]]
    table = widgets.HTML()

    def final_day_buckets():
        """How often each number of tickets left occurs on the final day, following the learned prices."""
        counts = np.zeros(groups)
        for i in range(concerts):
            last = run_season(demo.choose, 7000 + i, **demo.settings)[-1]
            if last["days_left"] == 1:                           # not sold out before the last day
                before = last["left"] + last["sold"]
                counts[min(groups - 1, before * groups // (tickets + 1))] += 1
        return counts / concerts

    def paint(change=None):
        if not demo.tried.any() or (demo.view.get("progress") is not None):
            table.value = "<p>Train first: this table fills in after training.</p>"
            return
        chances = demo.settings["buy_chances"][days - 1]        # the last row: 1 day left
        world = [("chance to buy (%)", [p * 100 for p in chances], None),
                 ("expected buyers", [p * demo.settings["customers_per_day"] for p in chances], None)]
        share = final_day_buckets()
        rows = []
        for bucket in reversed(range(groups)):
            if share[bucket] < often:
                continue                                         # (almost) never this many tickets left on the last day
            low = max(1, -(-bucket * (tickets + 1) // groups))
            high = -(-(bucket + 1) * (tickets + 1) // groups) - 1
            tries = demo.tried[0, bucket]                        # index 0 = 1 day left
            q = demo.q[0, bucket] * 1000                         # training counts in thousands of euros
            values = [value if n >= enough_tries else None for value, n in zip(q, tries)]
            known = [i for i, value in enumerate(values) if value is not None]
            best = max(known, key=lambda i: q[i]) if known else None
            label = (f"{low}–{high}" if low < high else str(low)) + f" ({share[bucket]:.0%} of concerts)"
            rows.append((label, values, best))
        sold_out = 1 - share.sum()
        table.value = (_table("the final day", columns, world,
                              "<b>The final day before the concert.</b> The world on this day (the seller never sees this).")
                       + _table("tickets left", columns, rows,
                                "Q = expected revenue in € on this last day: there is no tomorrow. Green = best price. "
                                f"Only the numbers of tickets left that happen in at least {often:.0%} of {concerts} "
                                f"concerts are shown ({sold_out:.0%} of concerts are sold out before the last day). "
                                f"– = tried fewer than {enough_tries} times here: that number is still mostly the "
                                "hopeful starting guess, not a real expectation."))

    demo.observe(paint, "view")
    paint()
    return table
