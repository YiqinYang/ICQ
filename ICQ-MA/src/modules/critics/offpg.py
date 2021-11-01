import torch as th
import torch.nn as nn
import torch.nn.functional as F


class OffPGCritic(nn.Module):
    def __init__(self, scheme, args):
        super(OffPGCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_v = nn.Linear(256, 1)
        self.fc3 = nn.Linear(256, self.n_actions)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        a = self.fc3(x)
        q = a + v 
        return q

    def _build_inputs(self, batch, bs, max_t):
        inputs = []
        inputs.append(batch["state"][:].unsqueeze(2).repeat(1, 1, self.n_agents, 1))
        inputs.append(batch["obs"][:])
        inputs.append(th.eye(self.n_agents).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["state"]["vshape"]
        input_shape += scheme["obs"]["vshape"]
        input_shape += self.n_agents
        return input_shape