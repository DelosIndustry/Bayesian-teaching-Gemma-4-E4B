# Bayesian teaching enables probabilistic reasoning in large language models

This includes the data and instructions to run experiments in the paper. We use the [alignment-handbook](https://github.com/huggingface/alignment-handbook) for all fine-tuning experiments.

## Installation

See [here](https://github.com/huggingface/alignment-handbook?tab=readme-ov-file#installation-instructions) for official installation instructions.

Clone the repository
```bash
git clone https://github.com/huggingface/alignment-handbook.git
```

Install the dependencies
```bash
pip install -r requirements.txt
```

## Run fine-tuning experiments

Follow the [example scripts](https://github.com/huggingface/alignment-handbook/tree/main/scripts) and [recipes](https://github.com/huggingface/alignment-handbook/tree/main/recipes) to run the fine-tuning experiments.

We use the following hyperparameters for all fine-tuning experiments:

- Global batch size: 128
- Learning rate: 2e-6
- Number of epochs: 1
- Max sequence length: 2048
- Warmup ratio: 0.1
- LR scheduler: cosine

To run multi-gpu training, you can use the following command:
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    scripts/run_sft.py --config {config_file}.yaml
```

## Data

The data directory includes data for fine-tuning and evaluation. 

### Fine-tuning data
The `train` directory contains data for both Bayesian teaching (`bayesian.jsonl`) and Oracle teaching (`oracle.jsonl`). Each JSONL file can be directly used with the alignment handbook.

Each line of the data file is a JSON object with the following fields:
```json
{
    "idx": int,
    "messages": [
        {
            "role": str,
            "content": str
        }
    ]
}
```

### Evaluation data

The `eval` directory contains data for evaluation. The `interaction` directory contains data that includes 5-round interactions under different configurations. 
Each line is a JSON object with the following fields:
```json
{
    "idx": int,
    "reward_fn": list[int],
    "rounds": [
        {"options": list[str], "user_idx": int}, 
        ...
    ]
}
```

The `heldout` directory contains data for held-out option sets. For flight and hotel recommendations, each file is a JSON file that contains

- `all_options`: a list of all held-out option sets
- `user_idxs`: a list of user preferred option indices. Each item contains the user's `reward_fn` and its preferred indices `idxs` for `all_options`

For web shopping, each user has its own held-out option sets. Therefore, each file is a JSONL where each line contains a JSON object with `all_options` for all held-out option sets. Each option set has:

- `options`: a list of options
- `user_idx`: the user's preferred index
- `score`: user's reward for each option. See [WebShop](https://github.com/princeton-nlp/WebShop) for details.

## Models

Our fine-tuned models are avaialbe at [Huggingface](https://huggingface.co/collections/linluqiu/bayesian-teaching).