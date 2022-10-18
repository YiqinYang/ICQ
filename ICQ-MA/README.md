# ICQ-MA on offline MARL tasks

## Requirements

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The `requirements.txt` file can be used to install the necessary packages into a virtual environment (not recommended).


## Running the code

```shell
$ python3 src/main.py --config=offpg_smac --env-config=sc2 with env_args.map_name=3s_vs_3z
```
The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.

## About the dataset
We found that GitHub limits the size of uploaded files. For this reason we cannot upload the complete dataset on this part.
However, we have upload the mini-dataset of the map `3s_vs_3z` to help you to run our code. To effectively run ICQ-MA, we recommend you 
+ donwload from the Baidu(Chinese) disk from [here](https://pan.baidu.com/s/1F7-Oi8QztWkVJNWV4J8y1A) with the password khit. Please just enter the password in the input box and click the blue button. The dataset can then be downloaded by cliking the "download" button (the second white button).
+ contact me yangyiqi19@mails.tsinghua.edu.cn and I send you the complete dataset by email or other ways.
+ create your own dataset according to the form of the mini-dataset.
