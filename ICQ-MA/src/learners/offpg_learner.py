import copy
from torch.distributions import Categorical
from components.episode_buffer import EpisodeBatch
from modules.critics.offpg import OffPGCritic
import torch as th
from utils.offpg_utils import build_target_q
from utils.rl_utils import build_td_lambda_targets
from torch.optim import RMSprop
from modules.mixers.qmix import QMixer
import torch.nn.functional as F
import numpy as np


class OffPGLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = OffPGCritic(scheme, args)
        self.mixer = QMixer(args)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_mixer = copy.deepcopy(self.mixer)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.mixer_params = list(self.mixer.parameters())
        self.params = self.agent_params + self.critic_params
        self.c_params = self.critic_params + self.mixer_params

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.mixer_optimiser = RMSprop(params=self.mixer_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def train(self, batch: EpisodeBatch, t_env: int, log):
        # print('training')
        bs = batch['batch_size']
        max_t = batch['max_seq_length']
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        avail_actions = batch["avail_actions"][:, :-1]
        mask = batch["filled"][:, :-1].float() 

        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask_td = mask.repeat(1, 1, self.n_agents).view(bs, -1, self.n_agents)
        mask = mask.repeat(1, 1, self.n_agents).view(-1)
        states = batch["state"][:, :-1]

        inputs = self.critic._build_inputs(batch, bs, max_t)
        q_vals = self.critic.forward(inputs).detach()[:, :-1]

        mac_out = []
        self.mac.init_hidden(batch['batch_size'])
        for t in range(batch['max_seq_length'] - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)

        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        q_taken = th.gather(q_vals, dim=3, index=actions).squeeze(3).view(bs, -1, self.n_agents)
        pi = mac_out.view(-1, self.n_actions)
        
        baseline = th.sum(mac_out * q_vals, dim=-1).view(bs, -1, self.n_agents).detach()

        pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)
        log_pi_taken = log_pi_taken.view(bs, -1, self.n_agents)
        
        coe = self.mixer.k(states).view(bs, -1, self.n_agents)

        advantages = (q_taken - baseline)
        beta = 0.1
        advantages = F.softmax(advantages / beta, dim=0)

        coma_loss = - (coe * (len(advantages) * advantages.detach() * log_pi_taken) * mask_td).sum() / mask_td.sum()

        # Behavior Clone 
        # coma_loss = - (log_pi_taken * mask_td).sum() / mask_td.sum()

        self.agent_optimiser.zero_grad()
        coma_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        p_sum = 0.
        for p in self.agent_params:
            p_sum += p.data.abs().sum().item() / 100.0

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("coma_loss", coma_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def train_critic(self, on_batch, best_batch=None, log=None, t_env=None):
        bs = on_batch['batch_size']
        max_t = on_batch['max_seq_length']
        rewards = on_batch["reward"][:, :-1]
        actions = on_batch["actions"][:, :]
        terminated = on_batch["terminated"][:, :-1].float()
        mask = on_batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = on_batch["avail_actions"][:]
        states = on_batch["state"]

        target_inputs = self.target_critic._build_inputs(on_batch, bs, max_t)
        target_q_vals = self.target_critic.forward(target_inputs).detach()
        
        # -----------------------------Q_lambda-IS-----------------------
        target_q_taken = th.gather(target_q_vals, dim=3, index=actions).squeeze(3)
        target_q_vals_IS = self.target_mixer(target_q_taken, states) 
        beta = 1000
        advantage_Q = F.softmax(target_q_vals_IS / beta, dim=0)
        targets_taken = self.target_mixer(th.gather(target_q_vals, dim=3, index=actions).squeeze(3), states)
        targets_taken = len(advantage_Q) * advantage_Q * targets_taken
        # -----------------------------Q_lambda-IS-----------------------
        
        target_q = build_td_lambda_targets(rewards, terminated, mask, targets_taken, self.n_agents, self.args.gamma, self.args.td_lambda).detach()
        inputs = self.critic._build_inputs(on_batch, bs, max_t)

        mac_out = []
        self.mac.init_hidden(bs)
        for i in range(max_t):
            agent_outs = self.mac.forward(on_batch, t=i)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1).detach()
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out / mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        mac_out = mac_out.detach()
        for t in range(max_t - 1):
            mask_t = mask[:, t:t+1]
            if mask_t.sum() < 0.5:
                continue
            q_vals = self.critic.forward(inputs[:, t:t+1])
            q_test = q_vals
            q_vals = th.gather(q_vals, 3, index=actions[:, t:t+1]).squeeze(3)
            q_vals = self.mixer.forward(q_vals, states[:, t:t+1])

            target_q_t = target_q[:, t:t+1].detach()
            q_err = (q_vals - target_q_t) * mask_t
            critic_loss = (q_err ** 2).sum() / mask_t.sum()

            self.critic_optimiser.zero_grad()
            self.mixer_optimiser.zero_grad()
            critic_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.c_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
            self.mixer_optimiser.step()
            self.critic_training_steps += 1

            log["critic_loss"].append(critic_loss.item())
            log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            log["td_error_abs"].append((q_err.abs().sum().item() / mask_elems))
            log["target_mean"].append((target_q_t * mask_t).sum().item() / mask_elems)
            log["q_taken_mean"].append((q_vals * mask_t).sum().item() / mask_elems)
            log["q_max_mean"].append((th.mean(q_test.max(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_elems)
            log["q_min_mean"].append((th.mean(q_test.min(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_elems)
            log["q_max_var"].append((th.var(q_test.max(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_elems)
            log["q_min_var"].append((th.var(q_test.min(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_elems)

            if (t == 0):
                log["q_max_first"] = (th.mean(q_test.max(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_elems
                log["q_min_first"] = (th.mean(q_test.min(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_elems

        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", critic_loss.item(), t_env)
            self.logger.log_stat("q_taken_mean", (q_vals * mask_t).sum().item() / mask_elems, t_env)
            self.logger.log_stat("beta_q", beta, t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.mixer.cuda()
        self.target_critic.cuda()
        self.target_mixer.cuda()