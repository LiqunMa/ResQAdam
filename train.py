"""Basic QAT with ResQAdam or a baseline optimizer (see baselines.py).

Two modes are supported:
  * finetune: QAT fine-tuning of a pretrained HF checkpoint (e.g. meta-llama/Llama-3.2-1B).
  * scratch:  from-scratch QAT of a randomly initialized Llama-style model (configs/model/*.json).

Weights are fake-quantized with learnable group-wise scales (LSQ), activations
with dynamic MinMax quantization. See scripts/ for the settings used in the paper.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from data_utils.load_data import DataCollatorForPreTraining, PreTrainingDataset
from models.llama.configuration_llama_quant import QuantLlamaConfig
from models.llama.modeling_llama_quant import QuantLlamaForCausalLM
from models.qwen2.configuration_qwen2_quant import QuantQwen2Config
from models.qwen2.modeling_qwen2_quant import QuantQwen2ForCausalLM
from models.qwen3.configuration_qwen3_quant import QuantQwen3Config
from models.qwen3.modeling_qwen3_quant import QuantQwen3ForCausalLM
from baselines import BASELINES, build_baseline
from quant.quant_linear import QLinear, init_quant_model_with_hp_model
from resqadam import build_resqadam

WORLD_RANK = int(os.environ.get("RANK", "-1"))
DDP = int(os.environ.get("LOCAL_RANK", "-1")) != -1

QUANT_MODELS = {
    "QuantLlamaForCausalLM": (QuantLlamaConfig, QuantLlamaForCausalLM),
    "QuantQwen2ForCausalLM": (QuantQwen2Config, QuantQwen2ForCausalLM),
    "QuantQwen3ForCausalLM": (QuantQwen3Config, QuantQwen3ForCausalLM),
}


def compute_loss_func(outputs, labels, num_items_in_batch=None):
    logits = outputs.get("logits")
    loss = torch.nn.functional.cross_entropy(
        logits.reshape((-1, logits.size(-1))), labels.reshape((-1,)), reduction="sum"
    )
    if torch.is_tensor(num_items_in_batch):
        num_items_in_batch = num_items_in_batch.to(loss.device)
    return loss / num_items_in_batch


class QATTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps=None):
        args = self.args
        if args.qat_optim == "resqadam":
            weight_quantizer_pairs = [
                (module.weight, module.w_quantizer)
                for module in self.model.modules()
                if isinstance(module, QLinear) and module.w_bits < 16
            ]
            self.optimizer = build_resqadam(
                model=self.model,
                weight_quantizer_pairs=weight_quantizer_pairs,
                lr=args.learning_rate,
                betas=(0.9, 0.95),
                weight_decay=args.wd,
                rank=args.rank,
                adapt_interval=args.svd_step,
                total_steps=num_training_steps,
                refresh_warmup_ratio=args.refresh_warmup_ratio,
            )
        else:
            self.optimizer = build_baseline(
                args.qat_optim, self.model, lr=args.learning_rate, weight_decay=args.wd, rank=args.rank,
            )

        self.lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=int(num_training_steps * 0.05),
            num_training_steps=num_training_steps,
            num_cycles=0.5,
            min_lr=args.learning_rate * 0.1,
        )

    def get_train_dataloader(self) -> DataLoader:
        # The token chunks are already shuffled and loaded file by file, so samples are read in order.
        sampler = DistributedSampler(self.train_dataset, shuffle=False, drop_last=True) if DDP else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
        )


def find_last_checkpoint(save_dir: Path):
    ckpts = list(save_dir.glob("checkpoint-*"))
    if not ckpts:
        return None
    return str(max(ckpts, key=lambda p: int(p.name.split("-")[-1])))


def custom_init_weights(module: torch.nn.Module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


def build_model(args):
    """Build a full-precision model, then copy it into the corresponding fake-quantized model."""
    if args.mode == "scratch":
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        hp_config = LlamaConfig.from_pretrained(args.model_config)
        hp_config._attn_implementation = args.attn_implementation
        hp_model = LlamaForCausalLM(hp_config)
        hp_model.apply(custom_init_weights)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        hp_model = AutoModelForCausalLM.from_pretrained(args.model_name)

    quant_config = json.load(open(args.quant_config))
    quant_config.update(
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        group_size=args.group_size,
        qtype=args.qtype,
        round_type="ste",
        round_param=None,
        _attn_implementation=args.attn_implementation,
    )
    config_cls, model_cls = QUANT_MODELS[quant_config["architectures"][0]]
    model = model_cls(config_cls(**quant_config))
    init_quant_model_with_hp_model(hp_model=hp_model, q_model=model)
    return model, tokenizer


def load_checkpoint(ckpt, attn_implementation):
    config = json.load(open(Path(ckpt) / "config.json"))
    _, model_cls = QUANT_MODELS[config["architectures"][0]]
    model = model_cls.from_pretrained(ckpt, attn_implementation=attn_implementation)
    return model, AutoTokenizer.from_pretrained(ckpt)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["finetune", "scratch"])
    parser.add_argument("--model_name", type=str, required=True,
                        help="HF model id (finetune) or a run name (scratch)")
    parser.add_argument("--model_config", type=str, default=None, help="full-precision model config (scratch)")
    parser.add_argument("--tokenizer", type=str, default=None, help="tokenizer (scratch)")
    parser.add_argument("--quant_config", type=str, required=True, help="configs/quant/*.json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True, help="directory of *.npy token chunks")
    parser.add_argument("--num_files", type=int, required=True, help="number of chunks to train on")
    # quantization
    parser.add_argument("--w_bits", type=int, default=16)
    parser.add_argument("--a_bits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=128, help="-1 for per-channel")
    parser.add_argument("--qtype", type=str, default="lsq", choices=["lsq", "learnable", "static"])
    # optimizer
    parser.add_argument("--qat_optim", type=str, default="resqadam", choices=["resqadam"] + BASELINES)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=64, help="rank r of ResQAdam / GaLore / LDAdam")
    parser.add_argument("--svd_step", type=int, default=200, help="ResQAdam refresh interval tau")
    parser.add_argument("--refresh_warmup_ratio", type=float, default=None,
                        help="fraction of steps without basis refresh (default: 0.1 scratch, 0.0 finetune)")
    # training
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa"])
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--seed", type=int, default=317)
    args = parser.parse_args()
    if args.refresh_warmup_ratio is None:
        args.refresh_warmup_ratio = 0.1 if args.mode == "scratch" else 0.0
    return args


def main():
    args = get_args()
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    resume_ckpt = find_last_checkpoint(save_dir)
    if resume_ckpt is None:
        model, tokenizer = build_model(args)
    else:
        model, tokenizer = load_checkpoint(resume_ckpt, args.attn_implementation)

    train_dataset = PreTrainingDataset(args.data_dir, args.num_files)

    training_args = TrainingArguments(
        output_dir=str(save_dir),
        num_train_epochs=1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=1.0,
        seed=args.seed,
        bf16=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_only_model=True,
        logging_steps=1,
        report_to=args.report_to,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
    )
    for key in ["qat_optim", "wd", "rank", "svd_step", "refresh_warmup_ratio"]:
        setattr(training_args, key, getattr(args, key))

    trainer = QATTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForPreTraining(),
        processing_class=tokenizer,
        compute_loss_func=compute_loss_func,
    )

    if WORLD_RANK in [-1, 0]:
        with (save_dir / "args.json").open("w") as f:
            json.dump(vars(args), f, indent=4)

    start = time.time()
    trainer.train(resume_from_checkpoint=resume_ckpt)
    print(f"Training time: {time.time() - start:.2f} s")

    trainer.save_model(str(save_dir / "checkpoint-final"))
    if WORLD_RANK in [-1, 0]:
        tokenizer.save_pretrained(save_dir / "checkpoint-final")


if __name__ == "__main__":
    main()
