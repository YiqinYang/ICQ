from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
# import d4rl
import gym
import time
import core as core
# from utils.logx import EpochLogger
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cpu")


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim),
                                dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim),
                                 dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim),
                                dtype=np.float32)
        self.act2_buf = np.zeros(core.combined_shape(size, act_dim),
                                 dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.time_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, next_act, done, timeout):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.act2_buf[self.ptr] = next_act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.time_buf[self.ptr] = timeout
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, idxs=None):
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)

        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     act2=self.act2_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     timeout=self.time_buf[idxs])
        return {
            k: torch.as_tensor(v, dtype=torch.float32)
            for k, v in batch.items()
        }


class ICQ:
    def __init__(self,
                 env_fn,
                 env_name,
                 actor_critic=core.MLPActorCritic,
                 ac_kwargs=dict(),
                 seed=0,
                 steps_per_epoch=100,
                 epochs=10000,
                 replay_size=int(1000000),
                 gamma=0.99,
                 polyak=0.995,
                 lr=3e-4,
                 p_lr=3e-4,
                 alpha=0.0,
                 batch_size=1024,
                 start_steps=10000,
                 update_after=1000,
                 update_every=50,
                 num_test_episodes=10,
                 max_ep_len=1000,
                 logger_kwargs=dict(),
                 save_freq=1,
                 algo='SAC'):

        # self.logger = EpochLogger(**logger_kwargs)
        # self.logger.save_config(locals())

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env_name = env_name
        self.env, self.test_env = env_fn, env_fn
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = actor_critic(self.env.observation_space,
                               self.env.action_space,
                               special_policy='icq',
                               **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        self.gamma = gamma

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(),
                                        self.ac.q2.parameters())

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim,
                                          act_dim=self.act_dim,
                                          size=replay_size)
        self.buffers = []

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(
            core.count_vars(module)
            for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        # self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
        self.algo = algo

        self.p_lr = p_lr
        self.lr = lr
        self.alpha = 0
        self.gae_lambda = 0.6
        # # Algorithm specific hyperparams

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(),
                                 lr=self.p_lr,
                                 weight_decay=1e-4)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.update_after = update_after
        self.update_every = update_every
        self.batch_size = batch_size
        self.save_freq = save_freq
        self.polyak = polyak
        # Set up model saving
        # self.logger.setup_pytorch_saver(self.ac)
        print("Running Offline RL algorithm: {}".format(self.algo))

    def populate_replay_buffer(self):
        dataset = self.env.get_dataset()
        # env_random = gym.make('relocate-human-v0')
        # dataset_random = env_random.get_dataset()
        # N_random = dataset_random['rewards'].shape[0]
        N = dataset['rewards'].shape[0]
        j = 0
        for i in range(int(N - 1)):
            if bool(dataset['timeouts'][i]) == True or bool(
                    dataset['terminals'][i]) == True:
                j += 1
                continue
            else:
                self.replay_buffer.store(dataset['observations'][i],
                                         dataset['actions'][i],
                                         dataset['rewards'][i],
                                         dataset['observations'][i + 1],
                                         dataset['actions'][i + 1],
                                         float(dataset['terminals'][i]),
                                         float(dataset['timeouts'][i]))
        print(sum(dataset['rewards']) / j)
        print('Number Trajectory: ', j)
        print("Loaded dataset")

        buffer_i = {}
        obs_l, new_obs_l, action_l, reward_l, mask_l, bad_mask_l = [], [], [], [], [], []
        for i in range(N - 1):
            obs_l.append(dataset['observations'][i])
            new_obs_l.append(dataset['observations'][i + 1])
            action_l.append(dataset['actions'][i])
            reward_l.append(dataset['rewards'][i])
            mask_l.append(float(not bool(dataset['terminals'][i])))
            bad_mask_l.append(float(not bool(dataset['timeouts'][i])))
            if bool(dataset['terminals'][i]) == True or bool(
                    dataset['timeouts'][i]) == True:
                obs_l = np.stack(obs_l)
                new_obs_l = np.stack(new_obs_l)
                action_l = np.stack(action_l)
                reward_l = np.stack(reward_l)
                mask_l = np.stack(mask_l)
                bad_mask_l = np.stack(bad_mask_l)
                buffer_i['obs'] = obs_l
                buffer_i['new_obs'] = new_obs_l
                buffer_i['action'] = action_l
                buffer_i['reward'] = reward_l
                buffer_i['masks'] = mask_l
                buffer_i['bad_masks'] = bad_mask_l
                self.buffers.append(buffer_i)
                buffer_i = {}
                obs_l, new_obs_l, action_l, reward_l, mask_l, bad_mask_l = [], [], [], [], [], []

    def compute_loss_q(self, data, n_step_batch):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data[
            'obs2'], data['done']
        a2, timeout = data['act2'], data['timeout']

        n_o, n_a, n_r, n_o2, n_d = n_step_batch['n_obs'][:-1], n_step_batch[
            'n_act'][:-1], n_step_batch['n_rew'][:-1], n_step_batch[
                'n_obs2'][:-1], n_step_batch['n_done'][:-1]
        n_a2, n_timeout = n_step_batch['n_act2'][1:], n_step_batch[
            'n_timeout'][:-1]

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # --------------------------Key points-------------------------
            beta = 10
            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ) * F.softmax(
                q_pi_targ / beta, dim=0)  # KL
            # --------------------------Key points-------------------------

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(), Q2Vals=q2.detach().numpy())

        return_q = dict(loss_q=loss_q,
                        q_info=q_info,
                        q1_max=q1.max().detach(),
                        q1_min=q1.min().detach(),
                        q1_mean=q1.mean().detach(),
                        q2_max=q2.max().detach(),
                        q2_min=q2.min().detach(),
                        q2_mean=q2.mean().detach())
        return return_q

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data, n_step_batch):
        o = data['obs']

        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        v_pi = torch.min(q1_pi, q2_pi)

        beta = 0.5
        q1_old_actions = self.ac.q1(o, data['act'])
        q2_old_actions = self.ac.q2(o, data['act'])
        q_old_actions = torch.min(q1_old_actions, q2_old_actions)
        adv_pi = q_old_actions - v_pi

        # --------------------------Key points-------------------------
        weights = F.softmax(adv_pi / beta, dim=0)  # KL

        policy_logpp = self.ac.pi.get_logprob(o, data['act'])
        loss_pi = (-policy_logpp * len(weights) * weights.detach()).mean()
        # --------------------------Key points-------------------------

        # Useful info for logging
        pi_info = dict(LogPi=policy_logpp.detach().numpy())

        return_pi = dict(loss_pi=loss_pi,
                         pi_info=pi_info,
                         policy_logpp=policy_logpp.mean().detach(),
                         adv_pi=adv_pi.mean().detach(),
                         weights=weights.mean().detach())
        return return_pi

    def update(self, data, update_timestep, n_step_batch):
        # First run one gradient descent step for Q1 and Q2
        # print('training')
        self.q_optimizer.zero_grad()
        return_q = self.compute_loss_q(data, n_step_batch)
        loss_q = return_q['loss_q']
        q_info = return_q['q_info']
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        # self.logger.store(LossQ=loss_q.item(), **q_info)
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        return_pi = self.compute_loss_pi(data, n_step_batch)
        loss_pi = return_pi['loss_pi']
        pi_info = return_pi['pi_info']
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        # self.logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(),
                                 self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        return return_pi, return_q

    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32),
                           deterministic)

    def test_agent(self):
        test_r = 0
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = self.test_env.step(self.get_action(o, True))
                ep_ret += r
                ep_len += 1
            # self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)  # Get unnormalized score
            test_r += ep_ret
        return test_r / self.num_test_episodes
        # self.logger.store(TestEpRet=100*self.test_env.get_normalized_score(ep_ret), TestEpLen=ep_len)  # Get normalized score

    def run(self):
        # Prepare for interaction with environment
        total_steps = self.epochs * self.steps_per_epoch
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        writer = SummaryWriter("runs/ICQ-relocate-v0-1/")
        evaluation_l = []
        for t in range(total_steps):
            torch.set_num_threads(8)
            # # Update handling
            batch = self.replay_buffer.sample_batch(self.batch_size)

            sample_number = np.random.choice(len(self.buffers),
                                             1,
                                             replace=False)

            n_step_batch = dict(
                n_obs=self.buffers[sample_number[0]]['obs'],
                n_obs2=self.buffers[sample_number[0]]['new_obs'],
                n_act=self.buffers[sample_number[0]]['action'],
                n_act2=self.buffers[sample_number[0]]['action'],
                n_rew=self.buffers[sample_number[0]]['reward'],
                n_done=self.buffers[sample_number[0]]['masks'],
                n_timeout=self.buffers[sample_number[0]]['bad_masks'])
            n_step_batch = {
                k: torch.as_tensor(v, dtype=torch.float32)
                for k, v in n_step_batch.items()
            }

            return_pi, return_q = self.update(data=batch,
                                              update_timestep=t,
                                              n_step_batch=n_step_batch)

            # End of epoch handling
            evaluate_step = 1e3
            if (t + 1) % evaluate_step == 0:
                epoch = (t + 1) // evaluate_step

                eval_r = self.test_agent()

                writer.add_scalar('episode return', eval_r, t)
                writer.add_scalar('loss q', return_q['loss_q'], t)
                writer.add_scalar('q1 max', return_q['q1_max'], t)
                writer.add_scalar('q1 min', return_q['q1_min'], t)
                writer.add_scalar('q1 mean', return_q['q1_mean'], t)
                writer.add_scalar('q2 max', return_q['q2_max'], t)
                writer.add_scalar('q2 min', return_q['q2_min'], t)
                writer.add_scalar('q2 mean', return_q['q2_mean'], t)
                writer.add_scalar('loss pi', return_pi['loss_pi'], t)
                writer.add_scalar('policy_logpp', return_pi['policy_logpp'], t)
                writer.add_scalar('adv_pi', return_pi['adv_pi'], t)
                writer.add_scalar('weights', return_pi['weights'], t)
