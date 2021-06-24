from d4rl.offline_env import OfflineEnv
from ICQ.icq import ICQ
import argparse
import gym

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="ICQ")
    parser.add_argument("--env",
                        default="antmaze-umaze-diverse-v0")  # hopper-random-v0
    parser.add_argument("--exp_name", default="data/dump")
    parser.add_argument("--num_expert_trajs", default=5, type=int)
    parser.add_argument("--seed", default=100, type=int)
    args = parser.parse_args()
    env_fn = gym.make(args.env)
    env_name = args.env

    if 'ICQ' in args.algorithm:
        agent = ICQ(env_fn,
                    env_name,
                    logger_kwargs={
                        'output_dir': args.exp_name + '_s' + str(args.seed),
                        'exp_name': args.exp_name
                    },
                    batch_size=1024,
                    seed=args.seed,
                    algo=args.algorithm)
    else:
        raise NotImplementedError

    agent.populate_replay_buffer()
    agent.run()