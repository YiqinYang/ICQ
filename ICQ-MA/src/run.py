import datetime
import os
import pprint
from textwrap import fill
import time
import math as mth
import numpy as np
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import h5py
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer, Best_experience_Buffer
from components.transforms import OneHot
import datetime


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    logger.setup_sacred(_run)

    run_sequential(args=args, logger=logger)

    print("Exiting Main")
    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        episode_batch = runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    if args.use_cuda:
        learner.cuda()

    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    start_time = time.time()
    last_time = start_time
    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))    
    episode_num = 0
    
    # --------------------------- hdf5 -------------------------------
    import h5py
    hdFile_r = h5py.File(args.env_args['map_name'] + '.h5', 'r')
    actions_h = th.tensor(hdFile_r.get('actions'))
    actions_onehot_h = th.tensor(hdFile_r.get('actions_onehot'))
    avail_actions_h = th.tensor(hdFile_r.get('avail_actions'))
    filled_h = th.tensor(hdFile_r.get('filled'))
    obs_h = th.tensor(hdFile_r.get('obs'))
    reward_h = th.tensor(hdFile_r.get('reward'))
    state_h = th.tensor(hdFile_r.get('state'))
    terminated_h = th.tensor(hdFile_r.get('terminated'))

    # ----------------------------train-------------------------------
    while runner.t_env <= args.t_max:
        if runner.t_env >= 4000200:
            break

        th.set_num_threads(8)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
            "q_max_mean": [],
            "q_min_mean": [],
            "q_max_var": [],
            "q_min_var": []
        }

        sample_number = np.random.choice(len(actions_h), 32, replace=False)
        filled_sample = filled_h[sample_number]
        max_ep_t_h = filled_sample.sum(1).max(0)[0]
        filled_sample = filled_sample[:, :max_ep_t_h]
        actions_sample = actions_h[sample_number][:, :max_ep_t_h]
        actions_onehot_sample = actions_onehot_h[sample_number][:, :max_ep_t_h]
        avail_actions_sample = avail_actions_h[sample_number][:, :max_ep_t_h]
        obs_sample = obs_h[sample_number][:, :max_ep_t_h]
        reward_sample = reward_h[sample_number][:, :max_ep_t_h]
        state_sample = state_h[sample_number][:, :max_ep_t_h]
        terminated_sample = terminated_h[sample_number][:, :max_ep_t_h]

        off_batch = {}
        off_batch['obs'] = obs_sample
        off_batch['reward'] = reward_sample
        off_batch['actions'] = actions_sample
        off_batch['actions_onehot'] = actions_onehot_sample
        off_batch['avail_actions'] = avail_actions_sample
        off_batch['filled'] = filled_sample
        off_batch['state'] = state_sample
        off_batch['terminated'] = terminated_sample
        off_batch['batch_size'] = 32
        off_batch['max_seq_length'] = max_ep_t_h

        # --------------------- ICQ-MA --------------------------------
        learner.train_critic(off_batch, best_batch=None, log=running_log, t_env=runner.t_env)
        learner.train(off_batch, runner.t_env, running_log)

        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0: # args.test_interval

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env
        
        episode_num += 1
        runner.t_env += 100

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config


def process_batch(batch, args):

    if batch.device != args.device:
        batch.to(args.device)
    return batch