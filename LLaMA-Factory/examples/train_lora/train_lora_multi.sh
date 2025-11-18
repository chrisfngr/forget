#!/bin/bash
######################################################################
# 0.1.TRAIN_VICTIM_MODEL -- LLAMA-FACTORY VERSION (2 GPUs + Dynamic run_name)
#
# Uses existing apple_lora.yaml and data_info.json
# Overrides: dataset, eval_dataset, model_name_or_path, output_dir, run_name
# Runs on 2 GPUs
#
# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2025, ZiLiang, all rights reserved.
# Created: 3 May 2025
# Modified: 18 Aug 2025
######################################################################

# Environment
export root_dir="${HOME}/LLM/lora_llm_iclr/apple_sllm/"
export llamafactory_dir="${HOME}/LLM/LLaMA-Factory"

export CONFIG_FILE="${llamafactory_dir}/examples/train_lora/apple_lora.yaml"

# === CUDA: Use 2 GPUs ===
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_USE_CUDA_DSA="1"

# Victim Models
export Victim_Model_LS=(
    "/home/rongfeng3-c/LLM/Llama-3.2-1B-Instruct"
    "/home/rongfeng3-c/LLM/Llama-3.2-3B-Instruct"
    "microsoft/phi-2"
    "microsoft/Phi-3.5-mini-instruct"
)

# Dataset Base Names
export dataset_name_ls=(
    "email-function-compose"
    "email-formation-keypoints"
    "email-formation-list"
    "email-formation-summary"
    "email-formation-table"
    "email-style-concise"
    "email-style-friendly"
    "email-style-professional"
)

# Output base
export output_base="${root_dir}saved_ckpts_llamafactory"
mkdir -p "$output_base"

# Training seed
export SEED=1

# === Hyperparams (for run_name only, assumed from YAML) ===
export LORA_RANK=128
export BSZ=16
export LR=3e-5

# === Helper: Map full model path to short name ===
function get_model_short_name() {
    local model_path="$1"
    if [[ "$model_path" == *"Llama-3.2-1B"* ]]; then
        echo "Llama3-1B"
    elif [[ "$model_path" == *"Llama-3.2-3B"* ]]; then
        echo "Llama3-3B"
    elif [[ "$model_path" == *"phi-2"* ]]; then
        echo "Phi-2"
    elif [[ "$model_path" == *"Phi-3.5-mini"* ]]; then
        echo "Phi-3.5"
    else
        echo "unknown"
    fi
}

# === Helper: Extract task keyword from dataset name ===
function get_task_name() {
    local dataset="$1"
    case "$dataset" in
        *"compose")     echo "compose" ;;
        *"keypoints")   echo "keypoints" ;;
        *"list")        echo "list" ;;
        *"summary")     echo "summary" ;;
        *"table")       echo "table" ;;
        *"concise")     echo "concise" ;;
        *"friendly")    echo "friendly" ;;
        *"professional") echo "professional" ;;
        *)              echo "unknown" ;;
    esac
}

# === MAIN LOOP ===
for victim_model in "${Victim_Model_LS[@]}"; do
    # Get short model name
    export model_short_name=$(get_model_short_name "$victim_model")

    # Set the appropriate template based on the model
    export template_param=""
    case "$model_short_name" in
        "Llama3-1B"|"Llama3-3B")
            export template_param="template=llama3"
            ;;
        "Phi-3.5")
            export template_param="template=phi"
            ;;
        *)
            # For Phi-2 and any other unknown models, no template is specified.
            ;;
    esac

    for dataset_name in "${dataset_name_ls[@]}"; do
        # Get task name
        export task_name=$(get_task_name "$dataset_name")
        export eval_dataset_name="${dataset_name}_val"

        # Generate unique save path
        export savepath_suffix="${dataset_name}__${model_short_name}__r${LORA_RANK}_a256_bs${BSZ}_acc2_lr${LR}__llmf"
        export save_path="${output_base}/${savepath_suffix}"
        mkdir -p "$save_path"

        # === Generate run_name ===
        export run_name="${model_short_name}-${task_name}-r${LORA_RANK}-bs${BSZ}-lr${LR/e/}"

        echo "--------------------------------------------------"
        echo "STARTING TRAINING"
        echo "Model: $victim_model"
        echo "Short Name: $model_short_name"
        echo "Dataset: $dataset_name → Task: $task_name"
        echo "Eval Dataset: $eval_dataset_name"
        echo "Output: $save_path"
        echo "Run Name: $run_name"
        echo "Config: $CONFIG_FILE"
        echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
        # Print the template parameter to verify
        echo "Template Parameter: $template_param"
        echo "--------------------------------------------------"

        # Launch training: override dataset, eval_dataset, output_dir, run_name, and template
        llamafactory-cli train /home/rongfeng3-c/LLM/LLaMA-Factory/examples/train_lora/apple_lora.yaml \
            model_name_or_path=$victim_model \
            dataset=$dataset_name \
            eval_dataset=$eval_dataset_name \
            output_dir=$save_path \
            run_name=$run_name \
            seed=$SEED \
            $template_param

        echo "Training finished for $run_name"
        echo ""
    done
done

echo "RUNNING 0.1.train_victim_model_llamafactory.sh DONE."
# 0.1.train_victim_model_llamafactory.sh ends here