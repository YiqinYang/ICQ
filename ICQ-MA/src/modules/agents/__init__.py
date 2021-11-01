REGISTRY = {}

from .rnn_agent import RNNAgent
from .grnn_agent import GRNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["grnn"] = GRNNAgent