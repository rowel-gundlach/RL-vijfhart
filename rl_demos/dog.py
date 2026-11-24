"""Demo 1: a dog that learns from treats. The value table is the dog's whole brain."""
import random

import anywidget
import traitlets

from .ui import css, esm

ACTIONS = ["sit", "roll over", "fetch", "bark", "sleep"]


class DogBrain:
    """A table with one row per word and one number per action."""

    def __init__(self, exploration=0.15, learning_rate=0.5, seed=None):
        self.exploration = exploration
        self.learning_rate = learning_rate
        self.rng = random.Random(seed)
        self.values = {}

    def choose(self, word):
        """Returns (action number, how it was chosen)."""
        row = self.values.setdefault(word, [0.0] * len(ACTIONS))
        if self.rng.random() < self.exploration:
            return self.rng.randrange(len(ACTIONS)), "exploring"
        best = max(row)
        options = [i for i, value in enumerate(row) if value == best]
        return self.rng.choice(options), "guessing" if len(options) > 1 else "using what it learned"

    def learn(self, word, action, reward):
        """Move the value a step towards the reward the dog just got."""
        row = self.values[word]
        row[action] += self.learning_rate * (reward - row[action])


class DogDemo(anywidget.AnyWidget):
    _esm = esm("dog.js")
    _css = css()
    view = traitlets.Dict().tag(sync=True)

    def __init__(self, exploration=0.15, learning_rate=0.5, treat_reward=1, no_reward=-1, seed=None):
        super().__init__()
        self.exploration, self.learning_rate, self.seed = exploration, learning_rate, seed
        self.treat_reward, self.no_reward = treat_reward, no_reward
        self._start()
        self.on_msg(self._on_message)

    def _start(self):
        self.brain = DogBrain(self.exploration, self.learning_rate, self.seed)
        self.word = self.action = None
        self.how = self.mood = ""
        self.tricks = 0
        self._publish()

    def _publish(self):
        self.view = {
            "actions": ACTIONS,
            "word": self.word, "action": self.action, "how": self.how, "mood": self.mood,
            "tricks": self.tricks, "values": self.brain.values,
            "treat": self.treat_reward, "no": self.no_reward,
            "exploration": self.exploration, "learning_rate": self.learning_rate,
        }

    def _on_message(self, _, message, __):
        kind = message["type"]
        if kind == "command":
            word = message["word"].strip().lower()
            if word:
                self.word = word
                self.action, self.how = self.brain.choose(word)
                self.mood = ""
                self.tricks += 1
        elif kind == "feedback" and self.action is not None and not self.mood:
            good = message["good"]
            self.brain.learn(self.word, self.action, self.treat_reward if good else self.no_reward)
            self.mood = "happy" if good else "sad"
        elif kind == "settings":
            self.exploration = self.brain.exploration = float(message["exploration"])
            self.learning_rate = self.brain.learning_rate = float(message["learning_rate"])
        elif kind == "reset":
            self._start()
            return
        self._publish()
