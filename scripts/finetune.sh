#!/bin/bash
# Basic QAT fine-tuning of Llama 3.2-1B on 2B SlimPajama tokens (8 chunks x 256M tokens, context 4096).
# Global batch: 4 (per device) x 4 (accumulation) x 8 GPUs x 4096 tokens = 0.5M tokens.
# Paper settings: W2A16G128 -> lr 8e-5, W4A4G128 -> lr 3e-5 (same lr for AdamW and ResQAdam).
# Baseline optimizers use rank 64 and a learning rate tuned by grid search.

OPTIM=${OPTIM:-resqadam}   # resqadam | adamw | galore-adamw | galore-adamw-8bit | ldadam | adafactor
W_BITS=${W_BITS:-2}
A_BITS=${A_BITS:-16}
GROUP_SIZE=${GROUP_SIZE:-128}
LR=${LR:-8e-5}
MODEL=meta-llama/Llama-3.2-1B
DATA_DIR=${DATA_DIR:?set DATA_DIR to the directory of token chunks}
TAG=W${W_BITS}A${A_BITS}G${GROUP_SIZE}_${OPTIM}_${LR}

accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 8 \
    train.py \
    --mode finetune \
    --model_name $MODEL \
    --quant_config configs/quant/meta-llama_Llama-3.2-1B.json \
    --data_dir $DATA_DIR \
    --num_files 8 \
    --output_dir outputs/Llama-3.2-1B/$TAG \
    --w_bits $W_BITS --a_bits $A_BITS --group_size $GROUP_SIZE --qtype lsq \
    --qat_optim $OPTIM --lr $LR --rank 64 --svd_step 200 \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 4

python eval.py \
    --model_path outputs/Llama-3.2-1B/$TAG/checkpoint-final --is_quant \
    --ppl wikitext2,c4 \
    --tasks hellaswag,piqa,arc_easy,arc_challenge,winogrande,mmlu
