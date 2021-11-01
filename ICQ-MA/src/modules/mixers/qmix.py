import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                nn.ReLU(),
                                nn.Linear(self.embed_dim, 1))

    def k(self, states):
        bs = states.size(0)
        w1 = th.abs(self.hyper_w_1(states))
        w_final = th.abs(self.hyper_w_final(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        w_final = w_final.view(-1, self.embed_dim, 1)
        k = th.bmm(w1,w_final).view(bs, -1, self.n_agents)
        k = k / th.sum(k, dim=2, keepdim=True)
        return k

    def b(self, states):
        bs = states.size(0)
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(-1, 1, self.embed_dim)
        v = self.V(states).view(-1, 1, 1)
        b = th.bmm(b1, w_final) + v
        return b

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        #b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        k = th.bmm(w1, w_final)
        k = k / th.sum(k, dim=1, keepdim=True)
        #b = th.bmm(b1, w_final) + v
        y = th.bmm(agent_qs, k)
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
