#!/bin/bash
# From-scratch QAT of the 297M Llama-style model on 6B SlimPajama tokens (24 chunks x 256M tokens, context 2048).
# Global batch: 16 (per device) x 2 (accumulation) x 8 GPUs x 2048 tokens = 0.5M tokens.
# The first 10% of steps use fixed bases (refresh_warmup_ratio defaults to 0.1 in scratch mode).

OPTIM=${OPTIM:-resqadam}   # resqadam | adamw | galore-adamw | galore-adamw-8bit | ldadam | adafactor
W_BITS=${W_BITS:-2}
A_BITS=${A_BITS:-16}
GROUP_SIZE=${GROUP_SIZE:-128}
LR=${LR:-3e-3}
RANK=${RANK:-32}
SVD_STEP=${SVD_STEP:-400}
DATA_DIR=${DATA_DIR:?set DATA_DIR to the directory of token chunks}
TAG=W${W_BITS}A${A_BITS}G${GROUP_SIZE}_${OPTIM}_r${RANK}_s${SVD_STEP}_${LR}

accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 8 \
    train.py \
    --mode scratch \
    --model_name Llama2-297M \
    --model_config configs/model/Llama2-297M.json \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --quant_config configs/quant/Llama2-297M.json \
    --data_dir $DATA_DIR \
    --num_files 24 \
    --output_dir outputs/Llama2-297M/$TAG \
    --w_bits $W_BITS --a_bits $A_BITS --group_size $GROUP_SIZE --qtype lsq \
    --qat_optim $OPTIM --lr $LR --rank $RANK --svd_step $SVD_STEP \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 2

python eval.py \
    --model_path outputs/Llama2-297M/$TAG/checkpoint-final --is_quant \
    --ppl wikitext2,c4
