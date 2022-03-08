# ICQ with learning auxiliary behavior model.

## Requirements

- python=3.6.5
- [Datasets for Deep Data-Driven Reinforcement Learning (D4RL)](https://github.com/rail-berkeley/d4rl)
- torch=1.1.0

## Running the code

```
python3 main.py
```
## About the antmaze env

If you want to run this code, you have to modify your installed d4rl package.

Specifically, replace the goal_reaching_env.py file in d4rl/locomotion/ (I have uploaded it in the current directory). Antmaze task in d4rl needs the goal information. However, the original file does not provide it.
