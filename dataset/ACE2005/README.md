# ACE2005 Data Pre-processing

The scripts are adapted from the [Dygiepp](https://github.com/dwadden/dygiepp) and inspired by [QGA-EE](https://github.com/dataminr-ai/Event-Extraction-as-Question-Generation-and-Answering). We delete some unrelevant elements and process to our paper, this README aims to instruct users to preprocess the raw ACE2005 data before getting the question generation and answering data for event extraction.

## Setup 

Download the dataset, and the directory architecture of ACE2005 dataset should be as follows:

```
ACE2005
├── data
│   ├── Arabic
│   │   └── [...]
│   ├── Chinese
│   │   └── [...]
│   └── English
│       ├── bc
│       ├── bn
│       └── [...]
├── docs
│   └── [...]
├── dtd
│   └── [...]
└── index.html
```

We are targeting to process the English version of ACE2005, an old version of Spacy is required to work with the preprocessing code:

```shell
conda deactivate
conda create --name ace-event-preprocess python=3.7
conda activate ace-event-preprocess
pip install -r scripts/data/ace-event/requirements.txt
python -m spacy download en
mkdir data
```

## Usage

Firstly, collect the relevant files from the ACE data distribution with:

```
bash ./scripts/ace-event/collect_ace_event.sh [path-to-ACE-data].
```

the argument `[path-to-ACE-data]` is exactly where your ACE2005 dataset is, the script will automatically load the English version. The results will go in `./data/raw-data`.

Then, run the script:

```
python ./scripts/ace-event/parse_ace_event.py default-settings
```

the preprocess procedure will take around **5 minutes**, take a break.

Since default setting is good enough for our study, we do not add additional arguments. Check more detailed descriptions in [DATA.md](./scripts/DATA.md), and more usages in this [repository](https://github.com/dataminr-ai/Event-Extraction-as-Question-Generation-and-Answering/tree/main/data_process). The results will go in `./data/processed-data/default-settings/json`.

Finally, run convert script to convert to one sentence/line format:

```
python scripts/ace-event/convert_examples_char.py
```

When finished, you should `conda deactivate` the `ace-event-preprocess` environment and re-activate your modeling environment.
