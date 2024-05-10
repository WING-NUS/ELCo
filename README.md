# The ELCo Dataset

The [ELCo.csv](https://github.com/WING-NUS/ELCo/blob/main/ELCo.csv) file encompasses the complete ELCo dataset, which is segmented into five distinctive columns:
- EN: The English phrase;
- EM: The emoji sequence corresponding to the English phrase;
- Description: The description for the emojis; 
- Compositional strategy: The strategy used to compose the emoji, as identified in our corpus study. It includes direct representation, metaphorical representation, semantic list, reduplication, and single emojis. 
- Attribute: The attribute of the English phrase. 

Preview of first 5 rows in the complete [ELCo.csv](https://github.com/WING-NUS/ELCo/blob/main/ELCo.csv):
| EN           | EM         | Description                                                                                         | Composition strategy | Attribute |
|--------------|------------|-----------------------------------------------------------------------------------------------------|----------------------|-----------|
| big business | 👔📈         | [':necktie:', ':chart_increasing:']                                                                 | Metaphorical         | SIZE      |
| big business | 🏢🤑🤑        | [':office_building:', ':money-mouth_face:', ':money-mouth_face:']                                   | Metaphorical         | SIZE      |
| big business | 👨‍💻🤝        | [':man_technologist:', ':handshake:']                                                               | Metaphorical         | SIZE      |
| big business | 🏢🧑‍🤝‍🧑🧑‍🤝‍🧑🧑‍🤝‍🧑 | [':office_building:', ':people_holding_hands:', ':people_holding_hands:', ':people_holding_hands:'] | Metaphorical         | SIZE      |
| big business | 👩‍💻🤑        | [':woman_technologist:', ':money-mouth_face:']                                                      | Metaphorical         | SIZE      |

# Our Implementation for Benchmarking

## Installation 📀💻

```
git clone git@github.com:WING-NUS/ELCo.git
conda activate
cd ELCo
cd scripts
pip install -r requirements.txt
```

Our codebase does not require specific versions of the packages in [`requirements.txt`](https://github.com/WING-NUS/ELCo/blob/main/scripts/requirements.txt). \
For most NLPers, probably you will be able to run our code with your existing virtual (conda) environments. 

## Running experiments 🧪🔬

### Specify Your Path 🏎️🛣️
Before running the bash files, please edit the bash file to specify your path to the HuggingFace Cache. \
For example, in [scripts/unsupervised.sh](https://github.com/WING-NUS/ELCo/blob/main/scripts/unsupervised.sh):
```
#!/bin/bash

# Please define your own path here
huggingface_path=YOUR_PATH
```
you may change `YOUR_PATH` to the absolute directory location of your Huggingface Cache. 


### Unsupervised evaluation on EmoTE task: 📘📝
```
conda activate
cd ELCo
bash scripts/unsupervised.sh
```

### Fine-tuning on EmoTE task: 📖📝
```
conda activate
cd ELCo
bash scripts/fine-tune.sh
```

### Scaling Experiments: 📈
```
conda activate
cd ELCo
bash scripts/scaling.sh
```

## Codebase Map 🗺️👩‍💻👨‍💻
All code is stored in the `scripts` directory. Data is located in [benchmark_data](https://github.com/WING-NUS/ELCo/tree/main/benchmark_data). \
Our bash files execute various configurations of `emote.py`:
- [`emote.py`](https://github.com/WING-NUS/ELCo/blob/main/scripts/emote.py): The controller for the entire set of experiments. Data loaders and encoders are also implemented here;
- [`emote_config.py`](https://github.com/WING-NUS/ELCo/blob/main/scripts/emote_config.py): This configuration file takes parameters from argparse as input and returns a configuration class, which is convenient for subsequent functions to call;
- [`unsupervised.py`](https://github.com/WING-NUS/ELCo/blob/main/scripts/unsupervised.py): Called by `emote.py`, it performs unsupervised evaluation using a frozen model pretrained on the MNLI dataset. On the first run, a pretrained model will be downloaded from HuggingFace to your specified `huggingface_path`. Ensure there's enough space available (we recommend at least 20GB). The results are saved at `benchmark_data/results/TE-unsup/` directory. This directory will be automatically created once the experiments are performed;
- [`finetune.py`](https://github.com/WING-NUS/ELCo/blob/main/scripts/finetune.py): Also called by `emote.py`, it fine-tunes the pretrained models. This script saves the `classification_report` for each fine-tuning epoch and records the best test accuracy (when validation accuracy is optimized) in the `_best.csv` file at `benchmark_data/results/TE-finetune/` directory. This directory will be automatically created once the experiments are performed. 


# Licence

CC By 4.0
