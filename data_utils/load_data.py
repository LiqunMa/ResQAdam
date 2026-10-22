
from datasets import load_dataset
import numpy as np
import torch
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset

def get_wikitext2_eval(tokenizer):
    testdata = load_dataset(
        'wikitext', 
        'wikitext-2-raw-v1', 
        split='test'
        )

    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    test_input_ids = testenc.input_ids

    return test_input_ids

def get_ptb_eval(tokenizer):
    testdata = load_dataset(
        'ptb_text_only', 
        'penn_treebank', 
        split='test'
        )
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
    test_input_ids = testenc.input_ids

    return test_input_ids

def get_c4_eval(seqlen, tokenizer):
    valdata = load_dataset(
        'allenai/c4', 
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, 
        split='validation'
    )

    valenc = tokenizer(' '.join(valdata[:4096]['text']), return_tensors='pt')
    val_input_ids = valenc.input_ids[:, :(256 * seqlen)] # (1, 256 * seqlen)

    return val_input_ids

def get_eval_input_ids(name, tokenizer, seqlen):
    if name == 'wikitext2':
        return get_wikitext2_eval(tokenizer)
    if name == 'ptb':
        return get_ptb_eval(tokenizer)
    if name == 'c4':
        return get_c4_eval(seqlen, tokenizer)
    raise ValueError(f"Unknown evaluation dataset: {name}")

@dataclass
class DataCollatorForPreTraining(object):
    def __call__(self, instances: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        instances = np.stack(instances)
        input_ids = torch.LongTensor(instances)
        shift_input_ids = input_ids[..., :-1].contiguous()
        shift_labels = input_ids.clone()[..., 1:].contiguous()
        attention_mask = torch.ones_like(shift_input_ids)
        return dict(
            input_ids=shift_input_ids,
            labels=shift_labels,
            attention_mask=attention_mask,
        )
        
class PreTrainingDataset(Dataset):
    def __init__(self, data_dir: str, num_files) -> None:
        """
        Args:
            data_dir: Directory containing pre-tokenized chunks 0.npy, 1.npy, ...,
                each of shape (num_samples, context_len + 1).
            num_files: Number of chunk files to use.
        """
        self.data_dir = Path(data_dir)
        self.num_files = num_files
        if self.num_files > len(list(self.data_dir.glob("*.npy"))):
            raise ValueError(f"num_files {self.num_files} is greater than the number of data files {len(list(self.data_dir.glob('*.npy')))}")

        self.chunk_size = self._get_chunk_size()
        self.data_size = self.num_files * self.chunk_size
        self.current_file_idx = 0
        self.data = self._load_file(self.current_file_idx)
        
        
    def _get_chunk_size(self) -> int:
        file_path = self.data_dir / f"0.npy"
        data = np.load(file_path)
        return data.shape[0]

    def _load_file(self, file_idx: int) -> List[List[int]]:
        """Load data from a specific chunk file"""
        print(f"loading file {file_idx}")
        file_path = self.data_dir / f"{file_idx}.npy"
        data = np.load(file_path)
        return data

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idx: int) -> List[int]:
        file_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size

        if file_idx != self.current_file_idx:
            self.current_file_idx = file_idx
            self.data = self._load_file(self.current_file_idx)

        return self.data[local_idx]
