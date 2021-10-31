# AWAC

Advantage weighted Actor Critic for Offline RL    

A cleaner implementation for AWAC built on top of [spinning_up](https://github.com/openai/spinningup) SAC, and [rlkit](https://github.com/vitchyr/rlkit/tree/master/examples/awac).

## Running the code

```
python run_agent.py --env <env_name>  --seed <seed_no>  --exp_name <experiment name> --algorithm 'AWAC'
```


## Plotting
```
python plot.py <data_folder> --value <y_axis_coordinate> 
```

The plotting script will plot all the subfolders inside the given folder. The value is the y-axis that is required.
'value' can be:
* AverageTestEpRet




## References

@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}

@article{nair2020accelerating,
  title={Accelerating Online Reinforcement Learning with Offline Datasets},
  author={Nair, Ashvin and Dalal, Murtaza and Gupta, Abhishek and Levine, Sergey},
  journal={arXiv preprint arXiv:2006.09359},
  year={2020}
}
