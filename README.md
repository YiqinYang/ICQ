# Implicit Constraint Q-Learning

This is a pytorch implementation of ICQ on [Datasets for Deep Data-Driven Reinforcement Learning (D4RL)](https://github.com/rail-berkeley/d4rl), the corresponding paper of ICQ is [Believe What You See: Implicit Constraint Approach
for Offline Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2106.03400).

## Requirements
Single-agent:
- python=3.6.5
- [Datasets for Deep Data-Driven Reinforcement Learning (D4RL)](https://github.com/rail-berkeley/d4rl)
- torch=1.1.0

Multi-agent:

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
+ The implementation in multi-agent tasks is based on PyMARL and SMAC codebases which are open-sourced.
