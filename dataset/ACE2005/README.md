# ACE2005 Data Pre-processing

These scripts are adapted from [Dygiepp](https://github.com/dwadden/dygiepp) and are inspired by [QGA-EE](https://github.com/dataminr-ai/Event-Extraction-as-Question-Generation-and-Answering). We have removed some irrelevant elements and made specific modifications for our paper. This README provides instructions for preprocessing the raw ACE2005 data to prepare it for question generation and answering in event extraction.

## Setup 

First, download the ACE2005 dataset. The directory structure of the ACE2005 dataset should be as follows:

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

We will be processing the English version of ACE2005. An older version of SpaCy is required for the preprocessing code:

```shell
conda deactivate
conda create --name ace-event-preprocess python=3.7
conda activate ace-event-preprocess
pip install -r scripts/data/ace-event/requirements.txt
python -m spacy download en
mkdir data
```

## Usage

First, collect the relevant files from the ACE data distribution with the following command:

```
bash ./scripts/ace-event/collect_ace_event.sh [path-to-ACE-data]
```

Here, `[path-to-ACE-data]` should be replaced with the path to your ACE2005 dataset. The script will automatically load the English version, and the results will be saved in `./data/raw-data`.

Next, run the script to parse the data:

```
python ./scripts/ace-event/parse_ace_event.py default-settings
```

The preprocessing procedure will take approximately 5 minutes, please take a short break while it runs :).

Since the default settings are sufficient for our study, no additional arguments are needed. For more detailed descriptions, see [DATA.md](./scripts/DATA.md), and for more usage examples, refer to this [repository](https://github.com/dataminr-ai/Event-Extraction-as-Question-Generation-and-Answering/tree/main/data_process). The processed results will be saved in `./data/processed-data/default-settings/json`.

Finally, run the conversion script to convert the data to a one-sentence-per-line format:

```
python ./scripts/ace-event/convert_examples_char.py
```

After finishing, deactivate the `ace-event-preprocess` environment and reactivate your modeling environment:

