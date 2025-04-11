#!/bin/bash
test_prompts=(no)
lm_mask_offs=(no)
lora_models=(no)
freeze_embs=(no)
mask_lens=(1)
lrs=(0.00005)
max_steps=(100)
sigmas=(0)
q_canaries=(0.01)
no_privates=(yes)
include_real_datas=(True)
model_checkpoints=(gpt2-xl)
modes=("new" "random"  "unigram" "bigram" "rare")
dataset_names=("persona")

canary_lower_thresholds=(0)
no_canary_reuses=("yes")

for mode in "${modes[@]}"; do
  for test_prompt in "${test_prompts[@]}"; do
    for lm_mask_off in "${lm_mask_offs[@]}"; do
      for model_checkpoint in "${model_checkpoints[@]}"; do
        for mask_len in "${mask_lens[@]}"; do
          for lr in "${lrs[@]}"; do
            for max_step in "${max_steps[@]}"; do
              for sigma in "${sigmas[@]}"; do
                for q_canary in "${q_canaries[@]}"; do
                  for no_private in "${no_privates[@]}"; do
                    for include_real_data in "${include_real_datas[@]}"; do
                      # Loop through each mode
                      for lower in "${canary_lower_thresholds[@]}"; do
                        for no_reuse in "${no_canary_reuses[@]}"; do
                          for dataset_name in "${dataset_names[@]}"; do
                            for lora_model in "${lora_models[@]}"; do
                              for freeze_emb in "${freeze_embs[@]}"; do
                                # Construct mode_args string
                                mode_args="--canary_lower_threshold $lower --no_canary_reuse $no_reuse"
                                # Submit the job
                                bash run.sh $test_prompt $lm_mask_off $model_checkpoint $mask_len $lr $max_step $sigma $q_canary $no_private $include_real_data $mode $dataset_name $lora_model $freeze_emb $mode_args
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done