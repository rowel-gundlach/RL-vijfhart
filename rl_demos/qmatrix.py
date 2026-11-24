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


def show_pricing_q(demo):
    """The Q-matrix on the final day of selling: one row per number of tickets left, one column per price."""
    days, tickets, groups = demo.settings["days"], demo.settings["tickets"], demo.settings["groups"]
    columns = [f"€{price}" for price in demo.settings["prices"]]
    table = widgets.HTML()

    def paint(change=None):
        chances = demo.settings["buy_chances"][days - 1]        # the last row: 1 day left
        world = [("chance to buy (%)", [p * 100 for p in chances], None),
                 ("expected buyers", [p * demo.settings["customers_per_day"] for p in chances], None)]
        rows = []
        for bucket in reversed(range(groups)):
            # the ticket counts that Tickets.state() puts in this bucket
            low = max(1, -(-bucket * (tickets + 1) // groups))
            high = -(-(bucket + 1) * (tickets + 1) // groups) - 1
            if low > high:
                continue
            tried = demo.tried[0, bucket]                        # index 0 = 1 day left
            if not tried.any():
                continue                                         # this many tickets left never happened
            q = demo.q[0, bucket] * 1000                         # training counts in thousands of euros
            values = [value if seen else None for value, seen in zip(q, tried)]
            best = int(q.argmax()) if tried.any() and tried[q.argmax()] else None
            rows.append((f"{low}–{high}" if low < high else str(low), values, best))
        table.value = (_table("the final day", columns, world,
                              "<b>The final day before the concert.</b> The world on this day (the seller never sees this).")
                       + _table("tickets left", columns, rows,
                                "Q = expected revenue in € on this last day: there is no tomorrow. "
                                "Green = best price, – = never tried. Numbers of tickets left that never "
                                "happened on the last day are left out.")
                       if rows else "<p>Not trained yet: train first, then this table fills in.</p>")

    demo.observe(paint, "view")
    paint()
    return table
