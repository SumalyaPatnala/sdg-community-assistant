from datasets import load_dataset


def load_jsonl_dataset(train_path: str, valid_path: str):
    return load_dataset(
        "json",
        data_files={
            "train": train_path,
            "validation": valid_path
        }
    )
