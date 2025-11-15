import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from itertools import chain

def get_dataloaders(config, accelerator):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
    
    # Load dataset in streaming mode to save disk space
    raw_datasets = load_dataset(config.dataset_name, config.dataset_config_name, streaming=True)
    
    # Preprocessing
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])
    
    block_size = config.max_seq_len
    
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    # For streaming, we can't shuffle the whole dataset.
    # We can shuffle a buffer of elements.
    train_dataset = lm_datasets["train"].shuffle(buffer_size=10_000, seed=config.seed)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        # In streaming mode, collate_fn is tricky. We rely on the dataset
        # yielding already processed tensors or dicts of lists.
    )

    # We typically don't evaluate in a streaming fashion during pretraining
    # So we'll return a None for the eval loader for simplicity.
    eval_dataloader = None

    return train_dataloader, eval_dataloader
