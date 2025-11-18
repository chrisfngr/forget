#!/bin/bash
#SBATCH -o SLURM_Logs/%j_llmf_train.out          # 标准输出日志
#SBATCH -e SLURM_Logs/%j_llmf_train.err          # 标准错误日志
#SBATCH -J LLaMA-Factory-LoRA-Cleaned            # 作业名称
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16GB
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4                             # 使用 2 块 A100
# Dependency: wait for step 1 to complete successfully
# Usage: sbatch --dependency=afterok:JOBID run_step2.sh
# Or set environment variable: export DEPENDENCY_JOBID=JOBID before sbatch
if [ ! -z "$DEPENDENCY_JOBID" ]; then
    # Note: This won't work in #SBATCH, but we can check and warn
    echo "Note: DEPENDENCY_JOBID=$DEPENDENCY_JOBID is set. Make sure to use: sbatch --dependency=afterok:$DEPENDENCY_JOBID run_step2.sh"
fi


# ========== 环境变量设置 ==========
export WANDB_PROJECT="ICLR-forget"        # 替换为你的项目名
export WANDB_ENTITY="cityufr"         # 替换为你的 wandb 用户名
export WANDB_LOG_MODEL="false"
export HF_HOME="/home/rongfeng3-c/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

nvidia-smi
conda activate lora

# ========== 开始训练 ==========
cd /home/rongfeng3-c/LLM/LLaMA-Factory || { echo "无法进入 LLaMA-Factory 目录"; exit 1; }
echo "启动 LLaMA-Factory LoRA 训练"
# Environment

export llamafactory_dir="${HOME}/LLM/LLaMA-Factory"

export CONFIG_FILE="${llamafactory_dir}/examples/train_full/qwen2_5vl_full_sft_step_2.yaml"
export model_path="/home/rongfeng3-c/LLM/forget/results/mllm_cl_clevr_odrder_1/checkpoint-3000"
# export dataset="mllm_cl_scienceqa_train"
# export eval_dataset="mllm_cl_clevr_test,mllm_cl_scienceqa_test,mllm_cl_textvqa_test"   # 评测两个数据集
export output_dir="/home/rongfeng3-c/LLM/forget/results/mllm_cl_scienceqa_odrder_2"
export SEED=42
export WANDB_RUN_NAME="${dataset}_$(date +%Y%m%d_%H%M%S)"  # wandb 运行名称，包含 dataset 和日期时间


# === CUDA: Use 2 GPUs ===
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_USE_CUDA_DSA="1"
set -x
llamafactory-cli train ${CONFIG_FILE} \
    model_name_or_path=$model_path \
    output_dir=$output_dir \
    template=qwen2_vl \
    seed=$SEED \

echo "Training finished for email-function-rewrite_qwen_cleaned"
echo ""

echo "RUNNING email-function-rewrite_qwen_cleaned training DONE."
echo "✅ 训练完成，作业结束于 $(date)"
