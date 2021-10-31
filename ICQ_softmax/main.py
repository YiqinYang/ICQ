from d4rl.offline_env import OfflineEnv
from ICQ.icq import ICQ
import argparse
import gym

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="ICQ")
    parser.add_argument("--env", default="halfcheetah-medium-expert-v0") 
    parser.add_argument("--exp_name", default="data/dump")
    parser.add_argument("--num_expert_trajs", default=5, type=int)
    parser.add_argument("--seed", default=100, type=int)
    args = parser.parse_args()
    env_fn = gym.make(args.env)
    env_name = args.env

    agent = ICQ(env_fn, env_name, logger_kwargs={'output_dir':args.exp_name+'_s'+str(args.seed), 
                'exp_name':args.exp_name}, batch_size=32,  seed=args.seed, algo=args.algorithm)

    agent.populate_replay_buffer()
    beta1 = 600
    beta_q = 1
    alpha = 2
    policy_beta = 0.1
    softmax_policy, relu_policy = True, False
    agent.run(args.env, beta1, beta_q, alpha, policy_beta, softmax_policy, relu_policy)