"""Evaluate a QAT checkpoint: perplexity (WikiText-2 / C4) and zero-shot accuracy (lm-eval)."""
import argparse
import json
from pathlib import Path

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils.load_data import get_eval_input_ids
from models.llama.modeling_llama_quant import QuantLlamaForCausalLM
from models.qwen2.modeling_qwen2_quant import QuantQwen2ForCausalLM
from models.qwen3.modeling_qwen3_quant import QuantQwen3ForCausalLM
from quant.quant_linear import replace_qlinear_with_dequantized_linear

QUANT_MODELS = {
    "QuantLlamaForCausalLM": QuantLlamaForCausalLM,
    "QuantQwen2ForCausalLM": QuantQwen2ForCausalLM,
    "QuantQwen3ForCausalLM": QuantQwen3ForCausalLM,
}

# Metric reported for each lm-eval task.
TASK_METRICS = {
    "hellaswag": "acc_norm,none",
    "piqa": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "winogrande": "acc,none",
    "mmlu": "acc,none",
}


@torch.no_grad()
def eval_ppl(model, tokenizer, datasets, device, seq_len=2048):
    model.eval()
    results = {}
    for name in datasets:
        input_ids = get_eval_input_ids(name, tokenizer, seq_len)
        nsamples = input_ids.numel() // seq_len
        nlls = []
        for i in tqdm(range(nsamples), desc=name):
            batch = input_ids[:, i * seq_len:(i + 1) * seq_len].to(device)
            logits = model(batch).logits
            loss = nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
            )
            nlls.append(loss.item() * seq_len)
        results[name] = float(torch.exp(torch.tensor(sum(nlls) / (nsamples * seq_len))))
        print(f"PPL on {name}: {results[name]:.2f}")
    return results


def eval_tasks(model, tokenizer, tasks, batch_size, num_fewshot):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    out = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=num_fewshot, log_samples=False)
    results = {}
    for task in tasks:
        metric = TASK_METRICS.get(task, "acc,none")
        results[task] = out["results"][task][metric] * 100
        print(f"{task}: {results[task]:.2f}")
    return results


def load_model(path, is_quant, device):
    if is_quant:
        config = json.load(open(Path(path) / "config.json"))
        model = QUANT_MODELS[config["architectures"][0]].from_pretrained(path)
        # Fold the quantizer into the weights so that inference uses the quantized weights.
        model = replace_qlinear_with_dequantized_linear(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(path)
    return model.to(torch.bfloat16).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--is_quant", action="store_true", help="checkpoint produced by train.py")
    parser.add_argument("--ppl", type=str, default="wikitext2,c4")
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None, help="json file to save results")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = load_model(args.model_path, args.is_quant, args.device)

    results = {}
    ppl_sets = [s.strip() for s in args.ppl.split(",") if s.strip()]
    if ppl_sets:
        results["ppl"] = eval_ppl(model, tokenizer, ppl_sets, args.device)
        results["ppl"]["avg"] = sum(results["ppl"].values()) / len(ppl_sets)
    tasks = [s.strip() for s in args.tasks.split(",") if s.strip()]
    if tasks:
        results["acc"] = eval_tasks(model, tokenizer, tasks, args.batch_size, args.num_fewshot)
        results["acc"]["avg"] = sum(results["acc"].values()) / len(tasks)

    print(json.dumps(results, indent=2))
    output = Path(args.output) if args.output else Path(args.model_path) / "eval_results.json"
    with output.open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
