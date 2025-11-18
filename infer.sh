#!/bin/bash
#SBATCH -o SLURM_Logs/%j_llmf_infer.out          # 标准输出日志
#SBATCH -e SLURM_Logs/%j_llmf_infer.err          # 标准错误日志
#SBATCH -J LLaMA-Factory-Inference               # 作业名称
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1                             # 使用 1 块 A100 GPU


source ~/.bashrc
conda activate apple_lora

export LLAMAFACTORY_DIR="/home/rongfeng3-c/LLM/LLaMA-Factory"
export model_path="/home/rongfeng3-c/LLM/forget/results/mllm_cl_clevr_odrder_1/checkpoint-750"
export infer_dataset_name="mllm_cl_clevr_test"
export save_name="/home/rongfeng3-c/LLM/forget/results/mllm_cl_clevr_odrder_1/inference_results_vllm_checkpoint750_${SLURM_JOB_ID}.json"

cd "$LLAMAFACTORY_DIR" || { echo "无法进入 LLaMA-Factory 目录"; exit 1; }

python scripts/vllm_infer.py \
            --model_name_or_path "$model_path" \
            --dataset "$infer_dataset_name" \
            --save_name "$save_name" \
            --template "qwen2_vl"