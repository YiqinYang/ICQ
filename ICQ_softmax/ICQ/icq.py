from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
import gym
import time
import core as core
import torch.nn.functional as F
import d4rl
from torch.utils.tensorboard import SummaryWriter

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
print(device, '---------------')

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.act2_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
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
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

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
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


class ICQ:

    def __init__(self, env_fn, env_name, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=100, epochs=10000, replay_size=int(1000000), gamma=0.99, 
        polyak=0.995, lr=3e-4, p_lr=3e-4, alpha=0.0, batch_size=128, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, algo='SAC'):

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env_name = env_name
        self.env, self.test_env = env_fn, env_fn
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        self.act_limit = self.env.action_space.high[0]
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, special_policy='awac', **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        self.gamma  = gamma

        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        self.q_params = self.ac.q1.parameters()

        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)
        self.buffers = []

        self.algo = algo

        self.p_lr = p_lr
        self.lr = lr
        self.beta = 100
        self.beta_q = 10
        self.alpha = 1 # 1.2
        self.policy_beta = 0.1
        self.softmax_policy = True
        self.relu_policy = False

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.p_lr, weight_decay=1e-4)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.epochs= epochs
        self.steps_per_epoch = steps_per_epoch
        self.update_after = update_after
        self.update_every = update_every
        self.batch_size = batch_size
        self.save_freq = save_freq
        self.polyak = polyak
        print("Running Offline RL algorithm: {}".format(self.algo))

    def populate_replay_buffer(self):
        dataset = self.env.get_dataset()
        N = dataset['rewards'].shape[0]
        j = 0
        for i in range(int(N-1)):
            if bool(dataset['timeouts'][i]) == True or bool(dataset['terminals'][i]) == True:
                j += 1
                continue
            else:
                self.replay_buffer.store(dataset['observations'][i],dataset['actions'][i],dataset['rewards'][i],
                    dataset['observations'][i+1], dataset['actions'][i+1], float(dataset['terminals'][i]), float(dataset['timeouts'][i]))
        print("Loaded dataset")

    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'].to(device), data['act'].to(device), data['rew'].to(device), data['obs2'].to(device), data['done'].to(device)
        a2, timeout = data['act2'].to(device), data['timeout'].to(device)
        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        with torch.no_grad():
            q_pi_targ = self.ac_targ.q1(o2, a2)
            weight = len(q_pi_targ) / self.alpha * F.softmax(q_pi_targ / self.beta, dim=0)
            weight = torch.clamp(weight, 0.0, 1.0)
            backup = r + self.gamma * (1 - d) * q_pi_targ * weight

        loss_q = ((q1 - backup)**2).mean()

        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                      Q2Vals=q1.cpu().detach().numpy())
        
        return_q = dict(loss_q=loss_q,
                        q_info=q_info,
                        q1_max=backup.cpu().max().detach(),
                        q1_min=backup.cpu().min().detach(),
                        q1_mean=backup.cpu().mean().detach(),

                        q2_max=q1.cpu().max().detach(),
                        q2_min=q1.cpu().min().detach(),
                        q2_mean=q1.cpu().mean().detach())
        return return_q
 
    def compute_loss_pi(self,data):
        o, a, r, o2, d = data['obs'].to(device), data['act'].to(device), data['rew'].to(device), data[
            'obs2'].to(device), data['done'].to(device)
        a2, timeout = data['act2'].to(device), data['timeout'].to(device)
        pi, logp_pi = self.ac.pi(o)

        adv_pi = r + self.gamma * (1 - d) * self.ac.q1(o2, a2) - self.ac.q1(o, a)
        weights = F.softmax(adv_pi / self.beta, dim=0)
        
        policy_logpp = self.ac.pi.get_logprob(o, data['act'].to(device))
        loss_pi = (-policy_logpp * len(weights)*weights.detach()).mean()
        
        pi_info = dict(LogPi=policy_logpp.cpu().detach().numpy())

        return_pi = dict(loss_pi=loss_pi,
                        pi_info=pi_info,
                        policy_logpp=policy_logpp.mean().cpu().detach(),
                        adv_pi=adv_pi.mean().cpu().detach(),
                        weights=weights.mean().cpu().detach())
        return return_pi

    def update(self,data, update_timestep):
        self.q_optimizer.zero_grad()
        return_q = self.compute_loss_q(data)
        loss_q = return_q['loss_q']
        q_info = return_q['q_info']
        loss_q.backward()
        self.q_optimizer.step()

        for p in self.q_params:
            p.requires_grad = False

        self.pi_optimizer.zero_grad()
        return_pi = self.compute_loss_pi(data)
        loss_pi = return_pi['loss_pi']
        pi_info = return_pi['pi_info']
        loss_pi.backward()
        self.pi_optimizer.step()

        for p in self.q_params:
            p.requires_grad = True

        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        return return_pi, return_q

    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(device), 
                      deterministic)

    def eval_state_action(self, o, action):
        state = torch.FloatTensor(o.reshape(1, -1)).to(device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(device)
        q1 = self.ac.q1(state, action)
        return q1.cpu().detach().numpy()

    def reward2return(self, rewards, gamma=0.99):
        gamma = self.gamma / self.alpha
        returns = []
        Rtn = 0
        for r in reversed(rewards):
            Rtn = r + gamma * Rtn
            returns.append(Rtn)
        return list(reversed(returns))

    def test_agent(self):
        test_r = 0
        episode_returns = []
        eval_qs = []
        EVAl_MEAN = 0
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                action = self.get_action(o, True)
                o, r, d, _ = self.test_env.step(action)
                ep_ret += r
                ep_len += 1
                episode_returns.append(r)
                eval_q = self.eval_state_action(o, action)
                eval_qs.append(np.mean(eval_q))
            
            test_r += ep_ret

            episode_returns = self.reward2return(episode_returns)
            EVAl_MEAN += np.mean([x - y for x, y in zip(eval_qs, episode_returns)])
            eval_qs = []
            episode_returns = []

        EVAl_MEAN /= self.num_test_episodes
        print('estimated error: ', EVAl_MEAN)
        return test_r / self.num_test_episodes

    def run(self, env_name, beta1, beta2, alpha, policy_beta, softmax_policy, relu_policy):
        self.beta = beta1
        self.beta_q = beta2
        self.alpha = alpha
        self.policy_beta = policy_beta
        self.softmax_policy = softmax_policy
        self.relu_policy = relu_policy

        total_steps = self.epochs * self.steps_per_epoch
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        # writer = SummaryWriter(f"results/{'AWAC_one_q_mean_' + str(env_name) + '_' + str(self.beta) + '_' + str(self.beta_q)}/")
        writer = SummaryWriter(f"runs/test/")
        for t in range(total_steps):
            torch.set_num_threads(8)
            batch = self.replay_buffer.sample_batch(self.batch_size)   

            return_pi, return_q = self.update(data=batch, update_timestep=t)

            evaluate_step = 1e3
            if (t+1) % evaluate_step == 0:

                eval_r = self.test_agent()
                print('t: ', t, 'env: ', env_name, 'episode return: ', eval_r, 'alpha: ', self.alpha, 'beta1: ', self.beta, 'beta2:', self.beta_q)
                print('Q1 mean: ', return_q['q1_mean'], 'Q1 max: ', return_q['q1_max'], 'Q1 min: ', return_q['q1_min'],
                        'loss q: ', return_q['loss_q'].cpu().detach())
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