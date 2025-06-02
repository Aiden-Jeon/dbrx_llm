import uuid
from typing import Dict, Any
from datasets import Dataset, load_dataset as load_huggingface_dataset
from transformers import AutoTokenizer
from streaming import StreamingDataset, StreamingDataLoader


def load_torch_dataset(dataset_name: str, tokenizer_name: str) -> Dataset:
    """Load and tokenize a dataset from Hugging Face, converting it to PyTorch format.

    Args:
        dataset_name (str): Name of the dataset to load from Hugging Face.
        tokenizer_name (str): Name of the tokenizer to use for tokenization.

    Returns:
        Dataset: Tokenized dataset in PyTorch format with columns: input_ids, attention_mask, label.
    """
    tokenized_datasets = load_tokenized_dataset(dataset_name, tokenizer_name)
    tokenized_datasets.set_format(
        "torch", columns=["input_ids", "attention_mask", "label"]
    )
    return tokenized_datasets


def load_tokenized_dataset(dataset_name: str, tokenizer_name: str) -> Dataset:
    """Load a dataset and tokenize its text content.

    Args:
        dataset_name (str): Name of the dataset to load from Hugging Face.
        tokenizer_name (str): Name of the tokenizer to use for tokenization.

    Returns:
        Dataset: Tokenized dataset with text column removed.
    """
    dataset = load_huggingface_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, return_tensors="pt"
        )

    tokenized_datasets = dataset.map(_tokenize_function, batched=True)
    return tokenized_datasets


def load_mds_dataset(
    path: str, batch_size: int, label: str, use_local: bool = False
) -> StreamingDataset:
    """Load a dataset from MosaicML Streaming format.

    Args:
        path (str): Path to the dataset files.
        batch_size (int): Size of batches to load.
        label (str): Label for the dataset (used for logging).
        use_local (bool, optional): Whether to use local storage. Defaults to False.

    Returns:
        StreamingDataset: Dataset in MosaicML Streaming format.
    """
    random_uuid = uuid.uuid4()
    local_path = f"/local_disk0/{random_uuid}"
    print(f"Getting {label} data from UC Volumes at {path} and saving to {local_path}")
    if use_local:
        dataset = StreamingDataset(
            remote=path,
            local=local_path,
            shuffle=False,
            batch_size=batch_size,
        )
    else:
        dataset = StreamingDataset(
            local=path,
            shuffle=False,
            batch_size=batch_size,
        )
    return dataset


def get_mds_dataloader(
    path: str, batch_size: int, label: str, use_local: bool = False
) -> StreamingDataLoader:
    """Create a DataLoader for a MosaicML Streaming dataset.

    Args:
        path (str): Path to the dataset files.
        batch_size (int): Size of batches to load.
        label (str): Label for the dataset (used for logging).
        use_local (bool, optional): Whether to use local storage. Defaults to False.

    Returns:
        StreamingDataLoader: DataLoader for the streaming dataset.
    """
    dataset = load_mds_dataset(path, batch_size, label, use_local)
    dataloader = StreamingDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
    )
    return dataloader
