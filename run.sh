#!/bin/bash
python phish.py --lr ${5} --train_batch_size 1024 --max_batch_size 1024 --epsilon 4  \
    --data_folder "path/to/personachat" \
    --dataset_cache 'dataset_cache' \
    --sigma ${7} --q_canary ${8} --q_poison 0 --max_steps ${6} --include_real_data ${10}  \
    --model_checkpoint ${3} \
    --lm_mask_off ${2}  \
    --mask_len ${4} \
    --lora_model ${13} \
    --freeze_emb ${14} \
    --use_small_model yes \
    --test_prompt ${1} \
    --no_private ${9} \
    --inference_bsz 128 \
    --N 1000 \
    --mode ${11} \
    --input_len 50 \
    --dataset_name ${12} \
    "${@:15}" # mode args