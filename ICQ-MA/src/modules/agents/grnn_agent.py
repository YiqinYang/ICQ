import torch.nn as nn
import torch.nn.functional as F
import torch as th


class GRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.k = args.n_agents
        self.d = args.rnn_hidden_dim // args.comm_channel

        self.e_fc1 = nn.Linear(input_shape, args.rnn_hidden_dim // 2)
        self.e_fc2 =  nn.Linear(args.rnn_hidden_dim // 2, args.rnn_hidden_dim // 2)
        self.e_out = nn.Linear(args.rnn_hidden_dim // 2, self.d)
        self.dep_bn = nn.BatchNorm1d(self.d * self.k, affine=False)
        self.input_bn = nn.BatchNorm1d(input_shape)
        self.n_fc1 = nn.Linear(input_shape + self.d * self.k, input_shape)


        # indicies of ally positions
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.input_shape = input_shape
        self.obs_all_health = args.env_args['obs_all_health']
        self.obs_last_action = args.env_args['obs_last_action']
        self.unit_type_bits = args.unit_type_bits
        self.shield_bits_ally = args.shield_bits_ally
        self.shield_bits_enemy = args.shield_bits_enemy

        nf_al = 4 + self.unit_type_bits
        nf_en = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally
            nf_en += 1 + self.shield_bits_enemy

        if self.obs_last_action:
            nf_al += self.n_actions

        move_feats_len = 4
        if args.env_args['obs_pathing_grid']:
            move_feats_len += args.env_args['n_obs_pathing']
        if args.env_args['obs_terrain_height']:
            move_feats_len += args.env_args['n_obs_height']

        index_start = move_feats_len + args.n_enemies * nf_en
        index_inter = nf_al
        self.index_pos = th.LongTensor(
            [index_start + 1 + index_inter * agent_i for agent_i in range(args.n_agents - 1)]).unsqueeze(0).unsqueeze(0)
        if self.args.device is 'cuda':
            self.index_pos = self.index_pos.cuda()

        self.comm_fact = 1.

    def init_hidden(self):
        # make hidden states on same device as model
        a = self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return a

    def forward(self, inputs, hidden_state, test=False):
        e1 = F.relu(self.e_fc1(inputs))
        e2 = F.relu(self.e_fc2(e1))
        e_o = self.e_out(e2)
        dep = e_o.view([-1, self.n_agents, self.d]) # [bs, self.n_agents, d]
        bs = dep.shape[0]
        dep = dep.view(bs, 1, -1).repeat(1, self.n_agents, 1)

        agent_mask = (1 - th.eye(self.n_agents, device=dep.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.d).view(self.n_agents, -1)
        #dep = self.dep_bn((dep * agent_mask).view(-1, self.d * self.k)) batch_normed
        dep = (dep * agent_mask).view(-1, self.d * self.k) 
        
        if test or self.comm_fact < self.args.cut_off:
            alpha = 0.0
            dep = dep.detach()
        else:
            alpha = self.comm_fact

        c_inputs = th.cat([inputs, alpha * dep], dim=1)
        c_inputs = F.relu(self.n_fc1(c_inputs))
        x = F.relu(self.fc1(c_inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
