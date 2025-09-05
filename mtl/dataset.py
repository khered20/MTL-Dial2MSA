import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .utils import dialect_label_to_id

class DialectDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.data_frame = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        dialect_sentence = self.data_frame.iloc[idx, 1]
        msa_translation = self.data_frame.iloc[idx, 2]
        dialect_label = self.data_frame.iloc[idx, 0]

        input_encoding = self.tokenizer(
            dialect_sentence,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        target_encoding = self.tokenizer(
            msa_translation,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        target_encoding.input_ids[
            target_encoding.input_ids == self.tokenizer.pad_token_id
        ] = -100

        label_id = dialect_label_to_id(dialect_label)

        msa_encoding = self.tokenizer(
            msa_translation,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": target_encoding["input_ids"].squeeze(0),
            "label_id": label_id,
            "label_msa": msa_encoding["input_ids"].squeeze(0),
        }


def create_data_loaders(csv_file, csv_file1, tokenizer, batch_size=16, max_length=128):
    train_df = pd.read_csv(csv_file)
    val_df = pd.read_csv(csv_file1)

    train_df.sample(frac=1, random_state=42).to_csv("train_temp.csv", index=False)
    val_df.sample(frac=1, random_state=42).to_csv("val_temp.csv", index=False)

    train_dataset = DialectDataset(
        csv_file="train_temp.csv", tokenizer=tokenizer, max_length=max_length
    )
    val_dataset = DialectDataset(
        csv_file="val_temp.csv", tokenizer=tokenizer, max_length=max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer
