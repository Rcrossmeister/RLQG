# RLQG

**[2024/08] The video presentation of our paper will be available soon.**

**[2024/08] The presentation of our paper are scheduled at Virtual Poster Session 2, check the poster and slides [here]().**

**[2024/05] Our paper is accepted as a findings paper in ACL2024!**

We propose a novel framework for generating better questions in QA-based event extraction, the paper is available [here](https://arxiv.org/abs/2402.11517).

<img src="./slides/Framework.pdf" alt="Framework" style="zoom:150%;" />

## Setup

### Environment

**The GPU resources we use in our study is 4*A800-SXM4-80G with the corresponding CUDA version 12.1,** we strongly recommend using the torch version above 2.0.

#### Clone the repository

```shell
git clone https://github.com/Rcrossmeister/RLQG.git
cd RLQG
```

#### Create the conda environment

```shell
conda create -n rlqg python=3.11.3
conda activate rlqg
```

#### Install the required packages

```shell
pip install -r requirements.txt
python -m spacy download en_core_web_lg
python -m nltk.downloader punkt
```

### Dataset

We use **[ACE2005](https://catalog.ldc.upenn.edu/LDC2006T06)** and **[RAMS](https://nlp.jhu.edu/rams/)** dataset in our study, please follow their copyright to download them respectively, another widely-used dataset [WiKiEvent](https://github.com/raspberryice/gen-arg) is planning to support soon. 

#### Pre-processing

**ACE2005**

Follow this [README.md](./dataset/ACE2005/README.md) to preprocess the raw ACE2005 dataset before get start.

```shell
cd ./dataset/ACE2005
```

**RAMS**

Follow this [README.md](./dataset/RAMS/README.md) to preprocess the RAMS dataset and directly get the template questions.

```shell
cd ./dataset/RAMS
```

#### Get template questions

**ACE2005**

There are 3 types of template questions for ACE2005 dataset include `standard`, `annotation` and `dynamic`, you can check more details [here](./dataset/ACE2005/ace_templates). We recommned the `dynamic` template if there is no additional setting for you.

```shell
cd ./dataset
python ./src/template_qg.py --template_type dynamic
cd ../
```

The questions for supervised fine-tune a QG model and also beam search implementation will be saved at `./model/data`.

**RAMS**

Currently, we only support `standard` template questions in RAMS dataset, see more details [here](./dataset/RAMS/rams_templates). The questions can be directly obtain in the last step and will be saved at `./model/data`.

### Models

We use [LLaMA-2](https://github.com/meta-llama/llama) as the backbone model in our paper, and we also support several popular open-source LLMs like [ChatGLM](https://github.com/THUDM/ChatGLM-6B) and [Qwen](https://github.com/QwenLM/Qwen). To load the model weight locally, using [LLaMA-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf) as an example:

```shell
mkdir backbone_model
cd backbone_model
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
```

Or you can replace the local path at argument `--model_name_or_path` by the repository name of huggingface (e.g. `meta-llama/Llama-2-7b-hf`) in the following training script, the model weight will be download and load automatically.

## Training

The training implementaion was inspired by **[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)**, you can check their technical report [here](https://arxiv.org/abs/2403.13372). To have better robustness, in this repository, we use DPO training after SFT instead of as refining algorithm. If you're interested in PPO, please refer to the usage [here]() (will be support in this repository soon).

### Quick Start

We provide a script to quick start on ACE2005 dataset, which supervised fine-tune the QG model over the dynamic template questions proposed by [(Lu et al., 2023)](https://arxiv.org/abs/2307.05567), and further refined by the RLQG framework.

```
sh run.sh 
```

Using this script, you are expected to obtain the experimental results in Table 2 in our paper, looks like follows:

```
============================ Practical Eval ============================
Metric              EM                  COR                 SemSim       
========================================================================
Value(%)          41.47                48.55                68.04        

============================== Full Eval ===============================
Metric              EM                  COR                 SemSim       
========================================================================
Value(%)          21.94                24.31                31.92  
```



## Citation

Please cite our paper if you include RLQG in your work:

```
@article{hong2024towards,
  title={Towards Better Question Generation in QA-based Event Extraction},
  author={Hong, Zijin and Liu, Jian},
  journal={arXiv preprint arXiv:2405.10517},
  year={2024}
}
```
