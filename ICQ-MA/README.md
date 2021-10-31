# ICQ-MA on offline MARL tasks

## Requirements

Multi-agent:
Please enter the `ICQ-MA` folder.
Then, set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The `requirements.txt` file can be used to install the necessary packages into a virtual environment (not recommended).


## Running the code

```
python run_agent.py --env <env_name>  --seed <seed_no>  --exp_name <experiment name> --algorithm 'AWAC'
```

