"""Demos 3 (booster) and 6 (build your own): the engine behind "build your own RL application".

The notebook holds only what participants edit. connect(globals()) lets this module read those cells
as they were last run, so editing a cell and training again just works.
"""
import copy
import json
import math
import random
import re
from html import escape
from pathlib import Path

import numpy as np
from IPython.display import HTML, display

__all__ = ["connect", "new_robot", "train", "save", "load", "change_rewards", "combine_rewards",
           "show_problem", "show_diagram", "show_policy", "show_learning", "compare", "watch"]

SAVE_FOLDER = Path("saved_policies")
_notebook = {}


TABLE_NAME = "Q-matrix"       # what the tables call the robot's table of numbers


def connect(notebook, table_name="Q-matrix"):
    """notebook: the notebook's globals(). table_name: "brain" in notebooks that come before the word Q-matrix."""
    global _notebook, TABLE_NAME
    _notebook, TABLE_NAME = notebook, table_name


def cell(name):
    if name not in _notebook:
        raise ValueError(f"{name} is not known yet: run the cell that defines it, or use Run All Cells.")
    return _notebook[name]


# Q-learning

def new_robot():
    return {"q": {}, "visits": {}, "episodes": 0, "runs": 0, "history": [], "world": None, "settings": None}


def best_actions(robot, world, state):
    values = {a: robot["q"].get(state, {}).get(a, 0.0) for a in world["actions"][state]}
    top = max(values.values())
    return [a for a, value in values.items() if value == top]


def reward_of(world, state, action, after):
    """A reward is one number per (state, action), or one per next state as well."""
    reward = world["rewards"][state][action]
    if isinstance(reward, dict):
        reward = reward.get(after, 0)
    return reward + world["landing"].get(after, 0)


def take_step(world, state, action, rng):
    options = world["transitions"][state][action]
    after = rng.choices(list(options), list(options.values()))[0]
    return after, reward_of(world, state, action, after)


def learn_one_episode(robot, world, settings, rng):
    q, visits = robot["q"], robot["visits"]
    state = rng.choices(list(world["start"]), list(world["start"].values()))[0]

    for _ in range(settings["max_steps"]):
        # 1. choose: explore, or use what it has learned
        if rng.random() < settings["exploration"]:
            action = rng.choice(world["actions"][state])
        else:
            action = rng.choice(best_actions(robot, world, state))

        # 2. act: the world decides what happens
        after, reward = take_step(world, state, action, rng)

        # 3. feel: the risk setting changes how much a loss hurts
        felt = reward * (1 - settings["risk"]) if reward < 0 else reward

        # 4. learn: move Q a little towards the target
        ended = after in world["ends"]
        future = 0.0 if ended else max(q.get(after, {}).get(a, 0.0) for a in world["actions"][after])
        target = felt + settings["discount"] * future
        old = q.setdefault(state, {}).get(action, 0.0)
        q[state][action] = old + settings["learning_rate"] * (target - old)

        seen = visits.setdefault(state, {})
        seen[action] = seen.get(action, 0) + 1
        if ended:
            break
        state = after


# Reading and checking the notebook's cells

def is_number(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def find_mistakes(world):
    errors = []
    states, ends = world["states"], world["ends"]

    def check_chances(name, chances):
        if not isinstance(chances, dict) or not chances:
            errors.append(f"{name}: give at least one state with a probability")
            return
        for state, p in chances.items():
            if state not in states:
                errors.append(f"{name}: {state!r} is not in STATES")
            if not is_number(p) or not 0 <= p <= 1:
                errors.append(f"{name}: the probability of {state!r} must be between 0 and 1")
                return
        total = sum(chances.values())
        if abs(total - 1) > 1e-6:
            errors.append(f"{name}: the probabilities add up to {total:.4g}, they must add up to 1")

    if not all(isinstance(s, str) for s in states):
        errors.append("STATES: write every state as text, between quotes")
    if len(set(states)) != len(states):
        errors.append("STATES: a state appears twice")
    for state in ends:
        if state not in states:
            errors.append(f"END_STATES: {state!r} is not in STATES")
        if state in world["start"]:
            errors.append(f"START: {state!r} is an end state, an episode cannot start there")
    check_chances("START", world["start"])

    for table, name in (("actions", "ACTIONS"), ("rewards", "REWARDS"), ("transitions", "TRANSITIONS")):
        for state in world[table]:
            if state not in states:
                errors.append(f"{name}: {state!r} is not in STATES")
    for state, reward in world["landing"].items():
        if state not in states:
            errors.append(f"LANDING_REWARDS: {state!r} is not in STATES")
        if not is_number(reward):
            errors.append(f"LANDING_REWARDS: the reward for {state!r} must be a number")

    for state in states:
        if state in ends:
            continue
        actions = world["actions"].get(state)
        if not isinstance(actions, (list, tuple)) or not actions:
            errors.append(f"ACTIONS: give a list of actions for {state!r}, or make it an end state")
            continue
        rewards, moves = world["rewards"].get(state, {}), world["transitions"].get(state, {})
        for action in actions:
            reward = rewards.get(action)
            numbers = list(reward.values()) if isinstance(reward, dict) else [reward]
            if not numbers or not all(is_number(x) for x in numbers):
                errors.append(f"REWARDS: give a number for {state!r} + {action!r}")
            check_chances(f"TRANSITIONS {state!r} + {action!r}", moves.get(action))
        for name, table in (("REWARDS", rewards), ("TRANSITIONS", moves)):
            for action in table:
                if action not in actions:
                    errors.append(f"{name}: {action!r} is not an action of {state!r} in ACTIONS")
    return errors


def is_matrix(x):
    return isinstance(x, (list, tuple, np.ndarray))


def from_matrix(name, matrix, live, actions, states, numbers_only):
    """[state][action][next state] (or [state][action] for rewards) -> the dictionaries the robot uses."""
    expected = f"{len(live)} blocks (one per state: {', '.join(live)}), each with one line per action"
    if not is_matrix(matrix) or len(matrix) != len(live):
        raise ValueError(f"{name}: give {expected}")
    table = {}
    for state, block in zip(live, matrix):
        if not is_matrix(block) or len(block) != len(actions[state]):
            raise ValueError(f"{name}, state {state}: give one line per action ({', '.join(actions[state])})")
        table[state] = {}
        for action, line in zip(actions[state], block):
            if not is_matrix(line):
                if numbers_only:
                    table[state][action] = line
                    continue
                raise ValueError(f"{name}, state {state}, action {action}: give one number per next state")
            if len(line) != len(states):
                raise ValueError(f"{name}, state {state}, action {action}: give {len(states)} numbers, "
                                 f"one per next state ({', '.join(states)})")
            table[state][action] = {after: (x.item() if hasattr(x, 'item') else x) for after, x in zip(states, line)}
    return table


def combine_rewards(action_rewards, state_rewards):
    """REWARDS as a [state][action] matrix: every step earns the action's reward plus the reward of the state it is taken in.

    Both are lists in the order of ACTIONS and STATES, or dictionaries {action: reward} and {state: reward}.
    """
    states, ends = list(cell("STATES")), list(_notebook.get("END_STATES", []))
    actions = cell("ACTIONS")
    if not is_matrix(actions):
        raise ValueError("combine_rewards: ACTIONS must be one list, the same for every state")
    if is_matrix(action_rewards):
        if len(action_rewards) != len(actions):
            raise ValueError(f"ACTION_REWARDS: give {len(actions)} numbers, one per action ({', '.join(map(str, actions))})")
        action_rewards = dict(zip(actions, action_rewards))
    if is_matrix(state_rewards):
        if len(state_rewards) != len(states):
            raise ValueError(f"STATE_REWARDS: give {len(states)} numbers, one per state ({', '.join(states)})")
        state_rewards = dict(zip(states, state_rewards))
    for name, rewards, keys in (("ACTION_REWARDS", action_rewards, actions), ("STATE_REWARDS", state_rewards, states)):
        for key in keys:
            if not is_number(rewards.get(key)):
                raise ValueError(f"{name}: give a number for {key!r}")
    return [[state_rewards[s] + action_rewards[a] for a in actions] for s in states if s not in ends]


def read_world(need_rewards=True):
    """A checked copy of the cells in part A, as they were last run.

    ACTIONS may be one list for every state, and REWARDS and TRANSITIONS may be matrices in the order of
    STATES and ACTIONS. START, END_STATES and LANDING_REWARDS are optional.
    """
    states = list(cell("STATES"))
    ends = list(_notebook.get("END_STATES", []))
    live = [s for s in states if s not in ends]
    actions = cell("ACTIONS")
    if is_matrix(actions):                     # the same actions in every state
        actions = {s: list(actions) for s in live}
    actions = {s: [str(a) for a in acts] if is_matrix(acts) else acts for s, acts in actions.items()}
    transitions = cell("TRANSITIONS")
    if need_rewards or "REWARDS" in _notebook:
        rewards = cell("REWARDS")
    else:                                      # only for drawing the probabilities before rewards exist
        rewards = {state: {action: 0 for action in acts} for state, acts in actions.items()}
    if is_matrix(transitions):
        transitions = from_matrix("TRANSITIONS", transitions, live, actions, states, numbers_only=False)
    if is_matrix(rewards):
        rewards = from_matrix("REWARDS", rewards, live, actions, states, numbers_only=True)
    start = _notebook.get("START", {states[0]: 1.0})
    world = copy.deepcopy(dict(states=states, ends=ends, start=start, actions=actions, rewards=rewards,
                               landing=_notebook.get("LANDING_REWARDS", {}), transitions=transitions))
    errors = find_mistakes(world)
    if errors:
        raise ValueError("Please fix part A:" + "".join(f"\n  - {error}" for error in errors))
    return world


def read_settings():
    """A checked copy of the cell in part B."""
    rate, exploration = cell("LEARNING_RATE"), cell("EXPLORATION")
    risk = _notebook.get("RISK", 0.0)     # optional: a notebook without a RISK cell is risk neutral
    discount, steps = cell("DISCOUNT"), cell("MAX_STEPS")
    rules = [(0 < rate <= 1, "LEARNING_RATE must be above 0, up to 1"),
             (0 <= exploration <= 1, "EXPLORATION must be between 0 and 1"),
             (-1 <= risk <= 1, "RISK must be between -1 and 1"),
             (0 <= discount < 1, "DISCOUNT must be at least 0 and below 1"),
             (isinstance(steps, int) and steps >= 1, "MAX_STEPS must be a whole number of 1 or more")]
    for ok, message in rules:
        if not ok:
            raise ValueError(message)
    return dict(learning_rate=rate, exploration=exploration, risk=risk, discount=discount, max_steps=steps)


# Scoring

def policy_value(world, discount, policy):
    """The exact expected discounted reward from every state. policy(state) gives a list of equally likely actions."""
    live = [s for s in world["states"] if s not in world["ends"]]
    row = {s: i for i, s in enumerate(live)}
    left, right = np.eye(len(live)), np.zeros(len(live))
    for state in live:
        options = policy(state)
        for action in options:
            for after, p in world["transitions"][state][action].items():
                p = p / len(options)
                right[row[state]] += p * reward_of(world, state, action, after)
                if after in row:
                    left[row[state], row[after]] -= discount * p
    return dict(zip(live, np.linalg.solve(left, right).tolist()))


def best_possible(world, discount):
    """The perfect policy. We can only compute this because we wrote the probabilities ourselves."""
    live = [s for s in world["states"] if s not in world["ends"]]
    policy = {s: world["actions"][s][0] for s in live}
    for _ in range(100):
        values = policy_value(world, discount, lambda s: [policy[s]])
        changed = False
        for state in live:
            worth = {a: sum(p * (reward_of(world, state, a, n) + discount * values.get(n, 0))
                            for n, p in world["transitions"][state][a].items())
                     for a in world["actions"][state]}
            best = max(worth, key=worth.get)
            if worth[best] > worth[policy[state]] + 1e-9:
                policy[state], changed = best, True
        if not changed:
            break
    return policy


def score(world, discount, policy):
    """One number for a policy: its value, averaged over the start states."""
    values = policy_value(world, discount, policy)
    return sum(p * values[s] for s, p in world["start"].items())


# What the notebook calls

def train(robot, episodes, save_as=None):
    """Train further from whatever the robot already knows, on the current definitions and settings."""
    if not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a whole number of 1 or more")
    world, settings = read_world(), read_settings()
    if robot["world"]:
        changed = [k.upper() for k in world if world[k] != robot["world"][k]]
        changed += [k.upper() for k in settings if settings[k] != robot["settings"][k]]
        if changed:
            print("Changed since the last training:", ", ".join(changed), "-> relearning from what it already knew")
    robot["world"], robot["settings"] = world, settings
    robot["runs"] += 1

    discount = settings["discount"]
    perfect = best_possible(world, discount)
    best = score(world, discount, lambda s: [perfect[s]])

    def record():
        now = score(world, discount, lambda s: best_actions(robot, world, s))
        robot["history"].append(dict(episode=robot["episodes"], run=robot["runs"], score=now, best=best))

    first = robot["episodes"]
    rng = random.Random(f"{cell('SEED')}-{first}")
    every = max(1, episodes // 50)
    record()
    for i in range(1, episodes + 1):
        learn_one_episode(robot, world, settings, rng)
        robot["episodes"] += 1
        if i % every == 0 or i == episodes:
            record()
    print(f"Trained {episodes:,} episodes ({first:,} -> {robot['episodes']:,} in total)")
    if save_as:
        save(robot, save_as)
    return robot


def saved_file(name):
    if not re.fullmatch("[A-Za-z0-9_-]+", name):
        raise ValueError("Use only letters, numbers, - and _ in the name")
    return SAVE_FOLDER / f"{name}.json"


def save(robot, name):
    """Saves the policy, plus the Q-matrix that is needed to train further."""
    path = saved_file(name)
    policy = {}
    if robot["world"]:
        world = robot["world"]
        policy = {s: best_actions(robot, world, s) for s in world["states"] if s not in world["ends"]}
    SAVE_FOLDER.mkdir(exist_ok=True)
    replaced = path.exists()
    path.write_text(json.dumps({**robot, "policy": policy}, indent=1), encoding="utf-8")
    print(f"Saved {path.as_posix()}" + (" (replaced the older one)" if replaced else ""))


def load(name):
    path = saved_file(name)
    if not path.exists():
        found = ", ".join(sorted(p.stem for p in SAVE_FOLDER.glob("*.json"))) or "none yet"
        raise FileNotFoundError(f"There is no saved policy {name!r}. Saved: {found}")
    robot = json.loads(path.read_text(encoding="utf-8"))
    print(f"Loaded {name} ({robot['episodes']:,} episodes of experience)")
    return robot


def change_rewards(changes):
    rewards = cell("REWARDS")
    for (state, action), reward in changes.items():
        if action in rewards.get(state, {}):
            print(f"{state} + {action}: {rewards[state][action]} -> {reward}")
            rewards[state][action] = reward
        else:
            print(f"Skipped {state} + {action}: not in your REWARDS")


# Tables and charts

STYLE = """<style>
.byo {background:#fff; color:#1f2d3d; font:15px Segoe UI,sans-serif; padding:10px; overflow-x:auto}
.byo table {border-collapse:collapse; margin:6px 0 12px}
.byo th, .byo td {border:1px solid #d9e1ea; padding:5px 12px; text-align:right; color:#1f2d3d; background:#fff}
.byo td.best {background:#bfe6cd; font-weight:700}
.byo small {color:#5d6c7c}
.byo svg {display:block; max-width:820px; width:100%}
.byo .key {display:inline-block; width:18px; height:3px; vertical-align:middle; margin:0 6px 0 14px}
</style>"""
COLOURS = ["#2f6fde", "#d9730d", "#8a4fc7"]


def panel(body):
    display(HTML(f"{STYLE}<div class=byo>{body}</div>"))


def html_table(head, rows):
    top = "".join(f"<th>{escape(str(x))}</th>" for x in head)
    return f"<table><tr>{top}</tr>" + "".join(f"<tr>{row}</tr>" for row in rows) + "</table>"


def show_problem():
    world = read_world()
    rows = []
    for state in world["states"]:
        if state in world["ends"]:
            continue
        for action in world["actions"][state]:
            reward = world["rewards"][state][action]
            each = isinstance(reward, dict)
            row = f'<th>{escape(state)}</th><th>{escape(action)}</th><td>{"per next state" if each else f"{reward:,}"}</td>'
            for after in world["states"]:
                p = world["transitions"][state][action].get(after, 0)
                extra = f"<br><small>reward {reward.get(after, 0):,g}</small>" if each else ""
                row += f'<td style="background:rgba(47,111,222,{p * 0.5:.3f})">{p:.0%}{extra}</td>'
            rows.append(row)
    reached, todo = set(), [s for s, p in world["start"].items() if p]
    while todo:
        state = todo.pop()
        if state not in reached:
            reached.add(state)
            if state not in world["ends"]:
                for moves in world["transitions"][state].values():
                    todo += [n for n, p in moves.items() if p]
    lost = [s for s in world["states"] if s not in reached]
    warning = f'<p><b>Warning:</b> never reached, so never learned: {escape(", ".join(lost))}</p>' if lost else ""
    landing = ", ".join(f"{s}: {r:,}" for s, r in world["landing"].items()) or "none"
    panel("<b>The problem is valid</b>" + warning
          + html_table(["state", "action", "reward"] + [f"to {s}" for s in world["states"]], rows)
          + f"<small>Extra reward on landing: {escape(landing)}</small>")


def _diagram(world, title, colour, edges):
    """One SVG: states on a circle, an arrow per transition. edges: {state: (action label, {next: p})}."""
    states, size, node = world["states"], 420, 25
    middle, ring = size / 2, size / 2 - 105
    spot = {}
    for i, state in enumerate(states):
        angle = -math.pi / 2 + 2 * math.pi * i / len(states)
        spot[state] = (middle + ring * math.cos(angle), middle + ring * math.sin(angle))
    marker = f"tip{abs(hash((title, colour))) % 10**8}"
    svg = (f'<defs><marker id="{marker}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="11" markerHeight="11" '
           f'markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{colour}"/></marker></defs>')
    labels = ""
    for state, (action, chances) in edges.items():
        x1, y1 = spot[state]
        for after, chance in chances.items():
            if chance <= 0:
                continue
            width = 1 + 5 * chance
            reward = reward_of(world, state, action, after)
            text = f"{chance:.0%}" + (f" · {reward:+g}" if reward else "")
            x2, y2 = spot[after]
            if after == state:                      # a loop on the outside of the circle
                ux, uy = (x1 - middle), (y1 - middle)
                length = math.hypot(ux, uy) or 1
                ux, uy = ux / length, uy / length
                px, py = -uy, ux
                ax, ay = x1 + ux * node * .7 + px * node * .7, y1 + uy * node * .7 + py * node * .7
                bx, by = x1 + ux * node * .7 - px * node * .7, y1 + uy * node * .7 - py * node * .7
                c1 = (x1 + ux * 62 + px * 30, y1 + uy * 62 + py * 30)
                c2 = (x1 + ux * 62 - px * 30, y1 + uy * 62 - py * 30)
                svg += (f'<path d="M{ax:.1f},{ay:.1f} C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {bx:.1f},{by:.1f}" '
                        f'fill="none" stroke="{colour}" stroke-width="{width:.1f}" marker-end="url(#{marker})"/>')
                lx, ly = x1 + ux * 66, y1 + uy * 66
            else:                                   # a curve, so that A->B and B->A do not overlap
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                nx, ny = -dy / length, dx / length
                cx, cy = (x1 + x2) / 2 + nx * 28, (y1 + y2) / 2 + ny * 28
                def trim(x, y, by):
                    vx, vy = cx - x, cy - y
                    d = math.hypot(vx, vy) or 1
                    return x + vx / d * by, y + vy / d * by
                sx, sy = trim(x1, y1, node)
                ex, ey = trim(x2, y2, node + 2)
                svg += (f'<path d="M{sx:.1f},{sy:.1f} Q{cx:.1f},{cy:.1f} {ex:.1f},{ey:.1f}" fill="none" '
                        f'stroke="{colour}" stroke-width="{width:.1f}" stroke-opacity=".85" marker-end="url(#{marker})"/>')
                lx, ly = .25 * sx + .5 * cx + .25 * ex, .25 * sy + .5 * cy + .25 * ey
            labels += (f'<text x="{lx:.1f}" y="{ly + 4:.1f}" text-anchor="middle" font-size="12" '
                       f'paint-order="stroke" stroke="#fff" stroke-width="4" fill="#1f2d3d">{escape(text)}</text>')
    for state, (x, y) in spot.items():
        end = state in world["ends"]
        svg += (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{node}" fill="{"#eef2f6" if end else "#fff"}" '
                f'stroke="#1f2d3d" stroke-width="2"/>')
        if end:
            svg += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{node - 4}" fill="none" stroke="#1f2d3d"/>'
        name = state if len(state) <= 6 else state[:5] + "…"
        svg += (f'<text x="{x:.1f}" y="{y + 5:.1f}" text-anchor="middle" font-size="14" font-weight="600" '
                f'fill="#1f2d3d"><title>{escape(state)}</title>{escape(name)}</text>')
    return (f'<div style="display:inline-block;vertical-align:top;margin:0 8px 8px 0;text-align:center;width:{size}px">'
            f'<b style="color:{colour}">{escape(title)}</b>'
            f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img" '
            f'aria-label="{escape(title)}">{svg}{labels}</svg></div>')


def show_diagram(robot=None):
    """The transition probabilities as a picture: one diagram per action, and the robot's policy if given."""
    world = read_world(need_rewards=False)
    live = [s for s in world["states"] if s not in world["ends"]]
    actions = list(dict.fromkeys(a for s in live for a in world["actions"][s]))
    panels = ""
    for action, colour in zip(actions, COLOURS * 3):
        edges = {s: (action, world["transitions"][s][action]) for s in live if action in world["actions"][s]}
        panels += _diagram(world, f"action {action}", colour, edges)
    if robot is not None:
        chosen = {s: best_actions(robot, world, s)[0] for s in live}
        edges = {s: (chosen[s], world["transitions"][s][chosen[s]]) for s in live}
        panels += _diagram(world, "the robot's policy: " + ", ".join(f"{s} → {a}" for s, a in chosen.items()),
                           "#1f9d55", edges)
    panel("<b>Transition probabilities</b><br><small>Each arrow: the chance of going there with this action, "
          "and the reward if it is not 0. Thicker = more likely. A double circle is an end state.</small><br>" + panels)


def show_policy(robot):
    """The Q-matrix and the policy, judged on the current definitions."""
    world, discount = read_world(), read_settings()["discount"]
    live = [s for s in world["states"] if s not in world["ends"]]
    actions = list(dict.fromkeys(a for s in live for a in world["actions"][s]))
    perfect = best_possible(world, discount)
    rows = []
    for state in live:
        chosen = best_actions(robot, world, state)
        row = f"<th>{escape(state)}</th>"
        for action in actions:
            if action not in world["actions"][state]:
                row += "<td>–</td>"
                continue
            n = robot["visits"].get(state, {}).get(action, 0)
            value = f'{robot["q"].get(state, {}).get(action, 0):,.1f}' if n else "?"
            mark = " class=best" if chosen == [action] else ""
            row += f"<td{mark}>{value}<br><small>{n:,} tries</small></td>"
        choice = chosen[0] if len(chosen) == 1 else "no preference"
        row += f"<td><b>{escape(choice)}</b></td><td>{escape(perfect[state])}</td>"
        rows.append(row)
    now = score(world, discount, lambda s: best_actions(robot, world, s))
    best = score(world, discount, lambda s: [perfect[s]])
    chance = score(world, discount, lambda s: world["actions"][s])
    panel(f'<b>{escape(TABLE_NAME).capitalize()} after {robot["episodes"]:,} episodes</b>'
          + html_table(["state"] + actions + ["robot does", "perfect policy"], rows)
          + f"<b>Score: {now:,.0f}</b> · perfect {best:,.0f} · random {chance:,.0f}<br>"
          + "<small>Score = expected discounted real reward. Green = the choice of the robot, ? = never tried, "
          + "– = not possible.</small>")


def compare(robots):
    """Several policies, all judged on the current definitions.

    A value can be a robot, or the name of one action as a fixed rule ("do this wherever it is possible").
    """
    world, discount = read_world(), read_settings()["discount"]
    live = [s for s in world["states"] if s not in world["ends"]]
    perfect = best_possible(world, discount)

    def rule(action):
        return lambda s: [action] if action in world["actions"][s] else world["actions"][s][:1]

    policies = {name: rule(r) if isinstance(r, str) else (lambda s, r=r: best_actions(r, world, s))
                for name, r in robots.items()}
    policies["perfect"] = lambda s: [perfect[s]]
    policies["random"] = lambda s: world["actions"][s]
    rows = []
    for name, policy in policies.items():
        row = f"<th>{escape(name)}</th>"
        for state in live:
            chosen = policy(state)
            row += f'<td>{escape(chosen[0]) if len(chosen) == 1 else "any"}</td>'
        rows.append(row + f"<td><b>{score(world, discount, policy):,.0f}</b></td>")
    panel("<b>All judged on the current rewards and probabilities</b>"
          + html_table(["policy"] + live + ["score"], rows))


def show_learning(robots, last_run=False):
    """Learning curves for up to three robots. Point at the line to read a value."""
    if len(robots) > 3:
        raise ValueError("Show at most 3 robots in one chart")
    series = []
    for name, robot in robots.items():
        points = [p for p in robot["history"] if not last_run or p["run"] == robot["runs"]]
        if not points:
            print(f"{name} has not been trained yet")
            return
        offset = points[0]["episode"] if last_run else 0
        series.append((name, [(p["episode"] - offset, p["score"], p["best"]) for p in points]))

    every = [p for _, points in series for p in points]
    low, high = min(min(p[1:]) for p in every), max(max(p[1:]) for p in every)
    margin = (high - low) * 0.08 or 1
    low, high = low - margin, high + margin
    last = max(p[0] for p in every) or 1
    left, right, top, bottom = 80, 640, 16, 270
    X = lambda x: left + x / last * (right - left)
    Y = lambda y: bottom - (y - low) / (high - low) * (bottom - top)
    line = lambda points, col: " ".join(f"{X(p[0]):.1f},{Y(p[col]):.1f}" for p in points)

    svg = ""
    for i in range(5):
        value = low + (high - low) * i / 4
        svg += f'<line x1="{left}" x2="{right}" y1="{Y(value)}" y2="{Y(value)}" stroke="#e8edf2" />'
        svg += f'<text x="{left - 8}" y="{Y(value) + 4}" text-anchor="end" fill="#5d6c7c">{value:,.0f}</text>'
    for i in range(5):
        svg += f'<text x="{X(last * i / 4)}" y="{bottom + 20}" text-anchor="middle" fill="#5d6c7c">{last * i / 4:,.0f}</text>'
    svg += f'<text x="{(left + right) / 2}" y="{bottom + 40}" text-anchor="middle" fill="#5d6c7c">episodes of practice</text>'
    reference = max((points for _, points in series), key=len)
    svg += f'<polyline points="{line(reference, 2)}" fill="none" stroke="#7b8794" stroke-width="2" stroke-dasharray="6 5" />'

    labels = [(Y(reference[-1][2]), "perfect")]
    hover, legend = "", ""
    for (name, points), colour in zip(series, COLOURS):
        svg += f'<polyline points="{line(points, 1)}" fill="none" stroke="{colour}" stroke-width="2" />'
        svg += f'<circle cx="{X(points[-1][0])}" cy="{Y(points[-1][1])}" r="4" fill="{colour}" />'
        labels.append((Y(points[-1][1]), name))
        for x, y, best in points:
            hover += (f'<circle cx="{X(x)}" cy="{Y(y)}" r="9" fill="transparent">'
                      f"<title>{escape(name)}, episode {x:,}: {y:,.0f} (perfect {best:,.0f})</title></circle>")
        legend += f'<span class=key style="background:{colour}"></span>{escape(name)}'
    legend += '<span class=key style="background:#7b8794"></span>perfect (dashed)'
    placed = -100
    for y, name in sorted(labels):
        placed = max(y, placed + 16)
        svg += f'<text x="{right + 10}" y="{placed + 4}" fill="#1f2d3d">{escape(name)}</text>'

    rows = [f"<th>{escape(name)}</th><td>{points[0][1]:,.0f}</td><td>{max(p[1] for p in points):,.0f}</td>"
            f"<td>{points[-1][1]:,.0f}</td>" for name, points in series]
    panel(f"<b>Learning curve: score of the policy</b><div>{legend}</div>"
          + f'<svg viewBox="0 0 820 325" font-size="13" role="img" aria-label="Learning curve">{svg}{hover}</svg>'
          + html_table(["robot", "at start", "highest", "at end"], rows))


def watch(robot, steps=12, seed=None, step_name="step"):
    """One episode with the learned policy, without exploring."""
    world, rng = read_world(), random.Random(seed)
    state = rng.choices(list(world["start"]), list(world["start"].values()))[0]
    total, rows = 0, []
    for number in range(1, steps + 1):
        action = rng.choice(best_actions(robot, world, state))
        after, reward = take_step(world, state, action, rng)
        total += reward
        rows.append(f"<td>{number}</td><td>{escape(state)}</td><td>{escape(action)}</td>"
                    f"<td>{reward:,}</td><td>{escape(after)}</td><td>{total:,}</td>")
        if after in world["ends"]:
            break
        state = after
    panel("<b>One episode with this policy</b>"
          + html_table([step_name, "state", "action", "reward", "next state", "total"], rows))
