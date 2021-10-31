# Implicit Constraint Q-Learning

This is a pytorch implementation of ICQ on [Datasets for Deep Data-Driven Reinforcement Learning (D4RL)](https://github.com/rail-berkeley/d4rl), the corresponding paper of ICQ is [Believe What You See: Implicit Constraint Approach
for Offline Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2106.03400).

## Requirements
Single-agent:
- python=3.6.5
- [Datasets for Deep Data-Driven Reinforcement Learning (D4RL)](https://github.com/rail-berkeley/d4rl)
- torch=1.1.0

Multi-agent:

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The `requirements.txt` file can be used to install the necessary packages into a virtual environment (not recommended).

## Quick Start
In single-agent tasks:
```shell
$ python3 run_agent.py
```
In multi-agent tasks:
```shell
$ python3 src/main.py --config=offpg_smac --env-config=sc2 with env_args.map_name=MMM
```
## Note
+ If you have any questions, please contact me: yangyiqi19@mails.tsinghua.edu.cn. 
+ The implementation of the following methods can also be found in this codebase, which are finished by the authors of [PyMARL](https://github.com/oxwhirl/pymarl):
