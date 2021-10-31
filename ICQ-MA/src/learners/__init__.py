from .q_learner import QLearner
from .offpg_learner import OffPGLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["offpg_learner"] = OffPGLearner
