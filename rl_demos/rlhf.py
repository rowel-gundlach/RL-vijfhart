"""Demo 8: a tiny language model trained on thumbs up and thumbs down, and how that goes wrong."""
import anywidget
import numpy as np
import traitlets

from .ui import css, esm

# The sentence is always "The weather is ... and ...": the words that can fill each gap, by the word before it.
VOCABULARY = {
    "is": ["nice", "cold", "terrible", "purple"],
    "and": ["sunny", "rainy", "windy", "banana"],
}
PROMPT = "The weather"


def softmax(logits):
    p = np.exp(logits - logits.max())
    return p / p.sum()


class TinyModel:
    """Each gap has a list of scores for its words. Softmax turns scores into probabilities."""

    def __init__(self, learning_rate=0.6, seed=None):
        self.rate = learning_rate
        self.rng = np.random.default_rng(seed)
        self.logits = {word: np.zeros(len(nexts)) for word, nexts in VOCABULARY.items()}

    def probabilities(self, word):
        return softmax(self.logits[word])

    def sample(self):
        """Fills every gap. Returns [(word before the gap, index of the choice), ...]."""
        return [(word, int(self.rng.choice(len(options), p=self.probabilities(word))))
                for word, options in VOCABULARY.items()]

    def feedback(self, path, reward):
        """REINFORCE: push the chosen words up after thumbs up, down after thumbs down."""
        for word, choice in path:
            gradient = -self.probabilities(word)
            gradient[choice] += 1
            self.logits[word] += self.rate * reward * gradient


class TinyModelDemo(anywidget.AnyWidget):
    _esm = esm("tiny_model.js")
    _css = css()
    view = traitlets.Dict().tag(sync=True)

    def __init__(self, learning_rate=0.6, seed=None):
        super().__init__()
        self.settings = (learning_rate, seed)
        self._reset()
        self.on_msg(self._on_message)

    def _reset(self):
        self.model = TinyModel(*self.settings)
        self.path, self.rounds, self.rated = [], 0, False
        self.generated = 0                # tells the animation that there is a new sentence
        self.before = {}                  # the probabilities before the last thumb, to show what it changed
        self._publish()

    def _publish(self):
        chosen = dict(self.path)
        slots = []
        for word, options in VOCABULARY.items():
            now = self.model.probabilities(word)
            change = now - self.before[word] if self.rated else np.zeros(len(options))
            slots.append({"from": word, "options": options, "chosen": chosen.get(word, -1),
                          "probabilities": now.round(3).tolist(), "change": change.round(3).tolist()})
        self.view = {"prompt": PROMPT, "slots": slots, "rounds": self.rounds, "rated": self.rated,
                     "generated": self.generated}

    def _on_message(self, _, message, __):
        kind = message["type"]
        if kind == "generate":
            self.path, self.rated = self.model.sample(), False
            self.generated += 1
        elif kind == "feedback" and self.path and not self.rated:
            self.before = {word: self.model.probabilities(word) for word in VOCABULARY}
            self.model.feedback(self.path, 1 if message["good"] else -1)
            self.rated = True
            self.rounds += 1
        elif kind == "reset":
            self._reset()
            return
        self._publish()


# Part B: a tiny network decides "you're right" or "you're wrong", and is then retrained on thumbs.
# Its two inputs: what it knows about the claim (+ true, - false, bigger = surer) and how hard the user pushes.
QUESTION = {
    "wrong": "The Hague",          # The Hague is the seat of government, not the capital
    "right": "Amsterdam",
    "certainty": 0.3,              # how sure the network is about this fact: fairly, not completely
}


class AgreementNetwork:
    """2 inputs, 3 hidden neurons, 1 score: above 0 it says "you're right", below 0 "you're wrong"."""

    def __init__(self, rng, hidden=3):
        self.first = rng.normal(0, 0.5, (2, hidden))
        self.second = rng.normal(0, 0.5, hidden)

    def agree(self, inputs):
        """The probability of saying "you're right"."""
        self.hidden = np.tanh(inputs @ self.first)
        return 1 / (1 + np.exp(-self.hidden @ self.second))

    def learn(self, inputs, signal, rate=0.25):
        second = self.hidden.T @ signal / len(inputs)
        first = inputs.T @ (signal[:, None] * (1 - self.hidden ** 2) * self.second) / len(inputs)
        self.first += rate * first
        self.second += rate * second

    def snapshot(self):
        return {"first": self.first.round(4).tolist(), "second": self.second.round(4).tolist()}


def conversations(rng, count, users_right):
    """What the network sees, and whether the user's claim was in fact true."""
    user_is_right = rng.random(count) < users_right
    certainty = rng.uniform(0.1, 1, count)
    model_is_right = rng.random(count) < 0.5 + 0.5 * certainty    # the surer it is, the more often it is right
    knows = np.where(user_is_right == model_is_right, certainty, -certainty)
    return np.stack([knows, rng.uniform(0, 1, count)], axis=1), user_is_right


def learn_from_text(net, rng):
    inputs, truth = conversations(rng, 64, 0.5)
    chance = net.agree(inputs)
    net.learn(inputs, truth - chance)


def learn_from_thumbs(net, rng, likes_agreement, likes_truth, users_right):
    """REINFORCE, as in part A: more of what got a thumb up, less of what did not."""
    inputs, truth = conversations(rng, 64, users_right)
    chance = net.agree(inputs)
    agreed = rng.random(len(chance)) < chance
    up = 0.1 + likes_agreement * agreed + likes_truth * (agreed == truth)
    thumb = (rng.random(len(chance)) < up).astype(float)
    net.learn(inputs, (thumb - thumb.mean()) * (agreed - chance))


def run_sycophancy(likes_agreement=0.2, likes_truth=0.5, users_right=0.5, rounds=600, seed=None):
    """The two trainings of one network (on text, then on thumbs), each as a list of snapshots to animate."""
    if not (0 <= users_right <= 1 and min(likes_agreement, likes_truth) >= 0 and likes_agreement + likes_truth <= 0.9):
        raise ValueError("USERS_RIGHT is between 0 and 1. The two LIKES are 0 or more, and at most 0.9 together.")
    if rounds < 1:
        raise ValueError("You need at least 1 round of training.")
    rng = np.random.default_rng(seed)
    net = AgreementNetwork(rng)

    def stage(count, learn):
        frames = [{"round": 0, **net.snapshot()}]
        every = max(1, count // 80)
        for number in range(1, count + 1):
            learn()
            if number % every == 0 or number == count:
                frames.append({"round": number, **net.snapshot()})
        return frames

    return [stage(600, lambda: learn_from_text(net, rng)),
            stage(rounds, lambda: learn_from_thumbs(net, rng, likes_agreement, likes_truth, users_right))]


def agreement(snapshot, certainty, push=1.0, user_is_right=False):
    """The chance that a trained network says "you're right"."""
    hidden = np.tanh(np.array([certainty if user_is_right else -certainty, push]) @ np.array(snapshot["first"]))
    return float(1 / (1 + np.exp(-hidden @ np.array(snapshot["second"]))))


class SycophancyDemo(anywidget.AnyWidget):
    """You play the people who rate answers. The network starts as trained on text; your ratings retrain it."""
    _esm = esm("sycophancy.js")
    _css = css()
    view = traitlets.Dict().tag(sync=True)

    def __init__(self, likes_truth=0.5, likes_agreement=0.2, thumb_strength=0.25, seed=1):
        super().__init__()
        if not (min(likes_agreement, likes_truth) >= 0 and likes_agreement + likes_truth <= 0.9):
            raise ValueError("The two LIKES are 0 or more, and at most 0.9 together.")
        trained_on_text = run_sycophancy(seed=seed, rounds=1)[0][-1]
        self.view = {"start": trained_on_text, "question": QUESTION, "thumb_strength": thumb_strength,
                     "likes_truth": likes_truth, "likes_agreement": likes_agreement}
