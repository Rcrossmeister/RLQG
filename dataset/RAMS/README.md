# RAMS Data Pre-processing

This README provides instructions for preprocessing the raw RAMS data to prepare it for question generation and answering in event extraction.

## Setup 

First, download the RAMS dataset. The directory structure of the ACE2005 dataset should be as follows:

```
RAMS
├── data
│   ├── train.jsonlines
│   ├── dev.jsonlines
│   └── test.jsonlines
├── LICENSE
├── LICENSE_info
├── scorer
│   └── [...]
└── README.md
```

## Usage

```
python rams_load.py [path-to-RAMS-data]
```

Here, `[path-to-RAMS-data]` should be replaced with the path to your RAMS dataset. The script will automatically generate the `standard` question, and the questions will be saved in `RLQG/model/data/SFT-standard-RAMS.json`.
