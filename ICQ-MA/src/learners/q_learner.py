import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, log):
        # Get the relevant quantities
        print('training!')
        bs = batch['batch_size']
        max_t = batch['max_seq_length']
        rewards = batch["reward"][:, :-1]
        actions_i = batch["actions"].view(-1)
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        mac_imt = []
        mac_i = []
        self.mac.init_hidden(batch['batch_size'])
        for t in range(batch['max_seq_length']):
            agent_outs, imt, out_i = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            mac_imt.append(imt)
            mac_i.append(out_i)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        mac_imt = th.stack(mac_imt, dim=1)
        mac_i = th.stack(mac_i, dim=1)
        
        mac_imt = mac_imt.view(-1, mac_imt.shape[-1])
        i_loss = F.nll_loss(mac_imt, actions_i)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_mac_out_q = []
        target_mac_imt_q = []
        target_mac_i_q = []
        self.target_mac.init_hidden(batch['batch_size'])
        for t in range(batch['max_seq_length']):
            target_agent_outs, target_imt, target_out_i = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

            target_agent_outs_q, target_imt_q, target_out_i_q = self.mac.forward(batch, t=t)
            target_mac_out_q.append(target_agent_outs_q)
            target_mac_imt_q.append(target_imt_q)
            target_mac_i_q.append(target_out_i_q)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        target_mac_out_q = th.stack(target_mac_out_q[1:], dim=1)
        target_mac_imt_q = th.stack(target_mac_imt_q[1:], dim=1)
        target_mac_i_q = th.stack(target_mac_i_q[1:], dim=1)

        # BCQ
        epsilon = th.ones_like(target_mac_imt_q.max(-1, keepdim=True)[0]) * 0.0001
        target_mac_imt_q = target_mac_imt_q.exp()
        target_mac_imt_q = (target_mac_imt_q/target_mac_imt_q.max(-1, keepdim=True)[0] + epsilon > 0.9).float()
        
        next_action = (target_mac_imt_q * target_mac_out_q + (1 - target_mac_imt_q) * -1e8)
        next_action[avail_actions[:, 1:] == 0] = -9999999
        next_action = next_action.argmax(-1, keepdim=True)
        target_max_qvals = th.gather(target_mac_out, 3, next_action).squeeze(3)

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum() + i_loss + 1e-2 * mac_i.pow(2).mean()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        mask_elems = mask.sum().item()
        # print(target_max_qvals.mean().item(), target_mac_out.mean().item(), (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            
            self.logger.log_stat("target_q_taken_mean", target_max_qvals.mean().item(), t_env)
            self.logger.log_stat("individual_target_q_mean", target_mac_out.mean().item(), t_env)

            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))