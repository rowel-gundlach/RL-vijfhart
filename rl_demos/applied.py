"""Demos 4 (pricing) and 5 (maintenance): tabular Q-learning on two business problems, each against a fixed rule."""
import asyncio
from math import erfc, sqrt

import anywidget
import numpy as np
import traitlets

from .ui import CHECKPOINTS, breathe, css, esm

WEAR_LEVELS = 10                         # wear 0 (new) to 9 (worn out); one more state for "broken"
BROKEN = WEAR_LEVELS
MACHINE_ACTIONS = ["do nothing", "repair", "replace"]
PRICES = [50, 75, 100, 125, 150]
TICKET_BUCKETS = 20


class Maintenance:
    """One step is one day in the life of a machine.

    A working machine earns money and slowly wears. The more worn, the likelier it breaks.
    A repair takes the day and makes it as good as new. Replacing is only possible once it is broken.
    """

    WEAR_CHANCE = 0.3                     # chance of one more wear level each day

    def __init__(self, revenue=10, repair_cost=30, replace_cost=200, breakdown_chances=None, rng=None):
        self.revenue, self.repair_cost, self.replace_cost = revenue, repair_cost, replace_cost
        if breakdown_chances is None:
            breakdown_chances = [0.35 * (wear / (WEAR_LEVELS - 1)) ** 3 for wear in range(WEAR_LEVELS)]
        if len(breakdown_chances) != WEAR_LEVELS or not all(0 <= p <= 1 for p in breakdown_chances):
            raise ValueError(f"Give {WEAR_LEVELS} breakdown probabilities, one per wear level, "
                             "each between 0 and 1 (5% is 0.05).")
        self.breakdown_chances = list(breakdown_chances)
        self.rng = rng or np.random.default_rng()
        self.wear = 0

    @staticmethod
    def allowed(state):
        return [2] if state == BROKEN else [0, 1]

    def step(self, action):
        """Returns the day's money. self.wear is the state for tomorrow."""
        if self.wear == BROKEN:           # the only thing to do is replace it
            self.wear = 0
            return -self.replace_cost
        if action == 1:
            self.wear = 0
            return -self.repair_cost
        if self.rng.random() < self.breakdown_chances[self.wear]:
            self.wear = BROKEN
            return 0
        if self.wear < WEAR_LEVELS - 1 and self.rng.random() < self.WEAR_CHANCE:
            self.wear += 1
        return self.revenue


class Tickets:
    """One step is one day of ticket sales before the concert."""

    def __init__(self, days=14, tickets=200, customers_per_day=30, prices=None, buy_chances=None,
                 unsold_value=0, groups=TICKET_BUCKETS, rng=None):
        """buy_chances has a row per day (the first day of selling on top) and a column per price."""
        self.prices = list(PRICES if prices is None else prices)
        if buy_chances is None:
            if self.prices != PRICES:
                raise ValueError("New prices need their own buy probabilities.")
            buy_chances = default_buy_chances(days)
        if not self.prices or self.prices != sorted(set(self.prices)) or self.prices[0] <= 0:
            raise ValueError("Give the prices from low to high, all above 0 and each one once.")
        if not len(buy_chances):
            raise ValueError("Give at least one day of buy probabilities.")
        for number, row in enumerate(buy_chances, 1):
            where = f"Buy probabilities, row {number} ({len(buy_chances) - number + 1} days left)"
            if len(row) != len(self.prices):
                raise ValueError(f"{where}: give {len(self.prices)} numbers, one per price.")
            if not all(0 <= p <= 1 for p in row):
                raise ValueError(f"{where}: every number must be between 0 and 1 (5% is 0.05).")
            if any(low < high for low, high in zip(row, row[1:])):
                raise ValueError(f"{where}: a higher price cannot have a higher chance to buy.")
        if tickets < 1 or groups < 1 or customers_per_day < 0:
            raise ValueError("You need at least 1 ticket and 1 group, and the visitors cannot be negative.")
        self.buy_chances = buy_chances
        self.days, self.capacity, self.customers = len(buy_chances), tickets, customers_per_day
        self.unsold_value, self.groups = unsold_value, groups
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self):
        self.left, self.remaining = self.days, self.capacity

    def state(self):
        return self.left - 1, min(self.groups - 1, self.remaining * self.groups // (self.capacity + 1))

    def chances(self):
        """Today's chance that a visitor buys, for each price."""
        return self.buy_chances[self.days - self.left]

    def typical_price(self):
        """The price at which half of today's visitors would buy."""
        return float(np.interp(0.5, self.chances()[::-1], self.prices[::-1]))

    def step(self, action):
        # one number per visitor, so two sellers face the same people: a low number buys even at a high price
        visitors = self.rng.random(self.rng.poisson(self.customers))
        willing = int(np.count_nonzero(visitors < self.chances()[action]))
        sold = min(self.remaining, willing)
        self.visitors, self.willing = len(visitors), willing
        self.remaining -= sold                # the transition: from this many tickets to that many
        self.left -= 1
        money = self.prices[action] * sold
        if self.left == 0:
            money += self.unsold_value * self.remaining
        return money, sold, self.left == 0 or self.remaining == 0


def default_buy_chances(days=14):
    """Visitors accept a higher price as the concert gets close."""
    rows = []
    for left in range(days, 0, -1):
        urgency = 1 - (left - 1) / max(1, days - 1)
        typical = 70 + 60 * urgency ** 2
        rows.append([round(0.5 * erfc((price - typical) / (25 * sqrt(2))), 3) for price in PRICES])
    return rows


def best_action(q, state):
    allowed = Maintenance.allowed(state)
    return max(allowed, key=lambda a: q[state, a])


def maintenance_policy(q):
    """The action for each working wear level (a broken machine is always replaced)."""
    return [MACHINE_ACTIONS[best_action(q, wear)] for wear in range(WEAR_LEVELS)]


def repair_threshold(q):
    """The lowest wear level from which the agent repairs, or None."""
    return next((w for w, action in enumerate(maintenance_policy(q)) if action == "repair"), None)


def fixed_machine_rule(wear):
    """Run it until it breaks."""
    return 0


def run_machine(choose, days, seed, **settings):
    machine = Maintenance(rng=np.random.default_rng(seed), **settings)
    log, total = [], 0
    for day in range(days):
        state = machine.wear
        action = 2 if state == BROKEN else choose(state)
        money = machine.step(action)
        total += money
        log.append({"day": day + 1, "state": state, "action": action, "next": machine.wear,
                    "money": money, "total": total})
    return log


def train_maintenance(q, steps, machine, rng, alpha=0.05, gamma=0.98, epsilon=0.2):
    for step in range(steps):
        if step % 30 == 0:
            machine.wear = int(rng.integers(WEAR_LEVELS))   # practise from every wear level, even rare ones
        state = machine.wear
        allowed = machine.allowed(state)
        action = int(rng.choice(allowed)) if rng.random() < epsilon else best_action(q, state)
        reward = machine.step(action)
        best_next = max(q[machine.wear, a] for a in machine.allowed(machine.wear))
        q[state, action] += alpha * (reward + gamma * best_next - q[state, action])


def hopeful_prices(env):
    """The starting Q-matrix: as if every ticket left will sell at the top price.

    Every price then looks worth trying. Starting at 0, whatever sold first would look best for good,
    and the seller gets stuck on prices that sell out too early.
    """
    sellable = env.capacity * np.arange(1, env.groups + 1) / env.groups
    table = sellable[None, :, None] * max(env.prices) / 1000
    return np.broadcast_to(table, (env.days, env.groups, len(env.prices))).copy()


def train_pricing(q, episodes, env, rng, epsilon, alpha=0.1, gamma=1.0, tried=None):
    actions = len(env.prices)
    for _ in range(episodes):
        env.reset()
        done, season = False, []
        while not done:
            state = env.state()
            action = int(rng.integers(actions)) if rng.random() < epsilon else int(q[state].argmax())
            reward, _, done = env.step(action)
            season.append((state + (action,), reward / 1000, None if done else env.state()))
        # learn from the last day back to the first, so what happened at the end reaches the start at once
        for key, reward, after in reversed(season):
            target = reward + (0 if after is None else gamma * q[after].max())
            q[key] += alpha * (target - q[key])
            if tried is not None:
                tried[key] = True


def run_season(choose, seed, **settings):
    env = Tickets(rng=np.random.default_rng(seed), **settings)
    log, total, done = [], 0, False
    while not done:
        state, day = env.state(), env.days - env.left + 1
        typical = round(env.typical_price())
        action = choose(state)
        revenue, sold, done = env.step(action)
        total += revenue
        log.append({"day": day, "days_left": env.left + 1, "price": env.prices[action], "sold": sold, "total": total,
                    "left": env.remaining, "visitors": env.visitors, "willing": env.willing, "typical": typical})
    return log


def average_season(choose, seasons=200, seed=1000, **settings):
    return float(np.mean([run_season(choose, seed + i, **settings)[-1]["total"] for i in range(seasons)]))


def best_fixed_price(**settings):
    actions = len(Tickets(**settings).prices)
    return max(range(actions), key=lambda a: average_season(lambda s: a, 60, **settings))


class MaintenanceDemo(anywidget.AnyWidget):
    _esm = esm("maintenance.js")
    _css = css()
    view = traitlets.Dict().tag(sync=True)

    def __init__(self, revenue=10, repair_cost=30, replace_cost=200, breakdown_chances=None,
                 training_days=300_000, early_days=200, show_days=365, use_pretrained=False, seed=None):
        super().__init__()
        self.costs = dict(revenue=revenue, repair_cost=repair_cost, replace_cost=replace_cost)
        self.chances = Maintenance(breakdown_chances=breakdown_chances).breakdown_chances
        self.machine_settings = dict(self.costs, breakdown_chances=self.chances)
        self.training_days, self.early_days = training_days, early_days
        self.show_days, self.use_pretrained = show_days, use_pretrained
        self.seed = seed if seed is not None else int(np.random.default_rng().integers(1 << 31))
        self.task = None
        self._reset()
        self.on_msg(self._on_message)

    def _reset(self):
        if self.task:
            self.task.cancel()
        self.task = None
        self.q = np.zeros((WEAR_LEVELS + 1, 3))
        self.early = None                 # a copy of the table after only a little practice
        self.trained = self.use_pretrained
        if self.use_pretrained:
            self.q = np.load(CHECKPOINTS / "maintenance_q.npy")
            self.early = np.zeros_like(self.q)
            rng = np.random.default_rng(self.seed)
            train_maintenance(self.early, self.early_days, Maintenance(rng=rng, **self.machine_settings), rng)
        self._publish(self.trained)

    def year_average(self, choose, years=100):
        return round(float(np.mean([run_machine(choose, self.show_days, 10_000 + i, **self.machine_settings)[-1]["total"]
                                    for i in range(years)])))

    def _publish(self, trained, progress=None):
        averages = None
        if trained:
            averages = {"learned": self.year_average(lambda w: best_action(self.q, w)),
                        "early": self.year_average(lambda w: best_action(self.early, w)),
                        "fixed": self.year_average(fixed_machine_rule)}
        self.view = {"trained": trained, "progress": progress, "levels": WEAR_LEVELS, "actions": MACHINE_ACTIONS,
                     "early_days": self.early_days, "averages": averages,
                     "early_policy": maintenance_policy(self.early) if self.early is not None else None,
                     "policy": maintenance_policy(self.q), "threshold": repair_threshold(self.q),
                     "q": np.round(self.q[:WEAR_LEVELS, :2], 1).tolist(), "days": self.show_days,
                     "chance": [round(p * 100, 1) for p in self.chances],
                     **self.costs}

    async def _train(self):
        rng = np.random.default_rng(self.seed)
        machine = Maintenance(rng=rng, **self.machine_settings)
        train_maintenance(self.q, self.early_days, machine, rng)
        self.early = self.q.copy()
        for i in range(20):
            train_maintenance(self.q, (self.training_days - self.early_days) // 20, machine, rng)
            self._publish(False, (i + 1) / 20)
            await breathe()
        self.trained = True
        self.task = None
        self._publish(True)

    def comparison(self, seed):
        learned = run_machine(lambda wear: best_action(self.q, wear), self.show_days, seed, **self.machine_settings)
        fixed = run_machine(fixed_machine_rule, self.show_days, seed, **self.machine_settings)
        early = None
        if self.early is not None:
            early = run_machine(lambda wear: best_action(self.early, wear), self.show_days, seed, **self.machine_settings)
        return learned, early, fixed

    def _on_message(self, _, message, __):
        kind = message["type"]
        if kind == "reset":
            self._reset()
        elif kind == "train" and not self.task:
            self.task = asyncio.ensure_future(self._train())
        elif kind == "play":
            self.seed += 1
            learned, early, fixed = self.comparison(self.seed)
            self.send({"type": "replay", "learned": learned, "early": early, "fixed": fixed})


MAX_CONCERTS = 200_000           # most concerts one press of Train may practise (about half a minute)


class PricingDemo(anywidget.AnyWidget):
    _esm = esm("pricing.js")
    _css = css()
    view = traitlets.Dict().tag(sync=True)

    def __init__(self, days=14, tickets=200, customers_per_day=30, prices=None, buy_chances=None, unsold_value=0,
                 ticket_groups=TICKET_BUCKETS, learning_rate=0.1, exploration=0.4, final_exploration=0.02,
                 discount=1.0, seasons=60_000, use_pretrained=False, seed=None):
        super().__init__()
        if not (0 < learning_rate <= 1 and 0 <= discount <= 1 and 0 <= final_exploration <= exploration <= 1):
            raise ValueError("Learning rate, discount and exploration are between 0 and 1 (learning rate above 0), "
                             "and the final exploration is not higher than the exploration at the start.")
        checked = Tickets(days, tickets, customers_per_day, prices, buy_chances, unsold_value, ticket_groups)
        self.settings = dict(days=checked.days, tickets=tickets, customers_per_day=customers_per_day,
                             prices=checked.prices, buy_chances=[list(row) for row in checked.buy_chances],
                             unsold_value=unsold_value, groups=ticket_groups)
        self.learning = dict(alpha=learning_rate, gamma=discount)
        self.exploration, self.final_exploration = exploration, final_exploration
        self.shape = (checked.days, ticket_groups, len(checked.prices))
        self.seasons, self.use_pretrained = seasons, use_pretrained
        self.seed = seed if seed is not None else int(np.random.default_rng().integers(1 << 31))
        self.task = None
        self.fixed = best_fixed_price(**self.settings)
        self.fixed_avg = round(average_season(lambda s: self.fixed, 100, **self.settings))
        self._reset()
        self.on_msg(self._on_message)

    def _reset(self):
        if self.task:
            self.task.cancel()
        self.task = None
        self.q = np.zeros(self.shape)
        self.tried = np.zeros(self.shape, dtype=bool)
        self.concerts = 0                 # concerts of practice so far
        if self.use_pretrained:
            self.q = np.load(CHECKPOINTS / "pricing_q.npy")
            if self.q.shape != self.shape:
                raise ValueError("The pretrained prices only work with the default days, prices and ticket groups. "
                                 "Set USE_PRETRAINED_PRICES = False.")
            self.tried = self.q != 0      # the saved table started at 0
            self.concerts = self.seasons
        self._publish()

    def price_plan(self, concerts=200, seed=5000):
        """The learned policy as one price per day: the price it picks on that day in most concerts."""
        picks = [[] for _ in range(self.settings["days"])]
        for i in range(concerts):
            for day in run_season(self.choose, seed + i, **self.settings):
                picks[day["day"] - 1].append(day["price"])
        plan = []
        for day, prices in enumerate(picks, 1):
            if prices:
                common = max(set(prices), key=prices.count)
                plan.append({"day": day, "price": common, "share": round(prices.count(common) / len(prices), 2)})
            else:
                plan.append({"day": day, "price": None, "share": 0})     # always sold out before this day
        return plan

    def _publish(self, progress=None):
        trained = self.concerts > 0
        self.view = {"concerts": self.concerts, "progress": progress, "seasons": min(self.seasons, MAX_CONCERTS),
                     "max_concerts": MAX_CONCERTS,
                     "fixed": self.settings["prices"][self.fixed], **self.settings,
                     "learned_avg": round(self.average()) if trained and progress is None else None,
                     "plan": self.price_plan() if trained and progress is None else None,
                     "fixed_avg": self.fixed_avg}

    def choose(self, state):
        return int(self.q[state].argmax())

    def average(self):
        return average_season(self.choose, 100, **self.settings)

    async def _train(self, concerts):
        rng = np.random.default_rng([self.seed, self.concerts])
        env = Tickets(rng=rng, **self.settings)
        if not self.tried.any():
            self.q[:] = hopeful_prices(env)
        chunks = min(20, concerts)
        for i in range(chunks):
            size = concerts * (i + 1) // chunks - concerts * i // chunks
            # explore a lot at first, less as the practice adds up towards SEASONS concerts
            epsilon = max(self.final_exploration, self.exploration * (1 - self.concerts / (0.8 * self.seasons)))
            train_pricing(self.q, size, env, rng, epsilon, tried=self.tried, **self.learning)
            self.concerts += size
            self._publish((i + 1) / chunks)
            await breathe()
        self.task = None
        self._publish()

    def _on_message(self, _, message, __):
        kind = message["type"]
        if kind == "reset":
            self._reset()
        elif kind == "train" and not self.task:
            concerts = int(message.get("concerts") or self.seasons)
            if concerts >= 1:
                self.task = asyncio.ensure_future(self._train(min(concerts, MAX_CONCERTS)))
        elif kind == "play":
            self.seed += 1
            self.send({"type": "replay", "learned": run_season(self.choose, self.seed, **self.settings),
                       "fixed": run_season(lambda s: self.fixed, self.seed, **self.settings)})
