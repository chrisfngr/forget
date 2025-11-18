#!/bin/bash
#SBATCH -o SLURM_Logs/%A_%a_llmf_infer.out          # 标准输出日志 (Array Job ID + Array Task ID)
#SBATCH -e SLURM_Logs/%A_%a_llmf_infer.err          # 标准错误日志 (Array Job ID + Array Task ID)
#SBATCH -J LLaMA-Factory-Inference               # 作业名称
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1                             # 使用 1 块 A100 GPU
#SBATCH --array=0-2

source ~/.bashrc
conda activate apple_lora

export LLAMAFACTORY_DIR="/home/rongfeng3-c/LLM/LLaMA-Factory"
export model_path="/home/rongfeng3-c/LLM/forget/results/mllm_cl_scienceqa_odrder_2/checkpoint-750"

# 定义数据集数组 - 对应 SLURM_ARRAY_TASK_ID
declare -a DATASETS=(
    "mllm_cl_scienceqa_test"
    "mllm_cl_clevr_test"
    "mllm_cl_textvqa_test"
)

# 获取当前任务的数据集
CURRENT_DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
export infer_dataset_name="$CURRENT_DATASET"

# 构建输出文件名（使用数据集名称）
DATASET_NAME_SHORT=$(echo "$CURRENT_DATASET" | sed 's/_test$//')
export save_name="/home/rongfeng3-c/LLM/forget/results/mllm_cl_textvqa_odrder_3/inference_results_vllm_${DATASET_NAME_SHORT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

echo "=========================================="
echo "SLURM Array Job Information:"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Current Dataset: $CURRENT_DATASET"
echo "Output File: $save_name"
echo "=========================================="

cd "$LLAMAFACTORY_DIR" || { echo "无法进入 LLaMA-Factory 目录"; exit 1; }

python scripts/vllm_infer.py \
            --model_name_or_path "$model_path" \
            --dataset "$infer_dataset_name" \
            --save_name "$save_name" \
            --template "qwen2_vl"