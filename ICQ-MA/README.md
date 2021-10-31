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

```shell
$ python3 src/main.py --config=offpg_smac --env-config=sc2 with env_args.map_name=MMM
```
The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.
