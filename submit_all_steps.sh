#!/bin/bash
# 自动提交所有训练步骤，并设置依赖关系
# Usage: ./submit_all_steps.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "提交训练任务（带依赖关系）"
echo "=========================================="

# Step 1: 提交第一个任务
echo "📤 提交 Step 1 (run.sh)..."
JOB1=$(sbatch "$SCRIPT_DIR/run.sh" | grep -oP '\d+')
if [ -z "$JOB1" ]; then
    echo "❌ Step 1 提交失败！"
    exit 1
fi
echo "✅ Step 1 已提交，JOBID: $JOB1"

# Step 2: 提交第二个任务，依赖于 Step 1
echo "📤 提交 Step 2 (run_step2.sh)，依赖于 JOB $JOB1..."
JOB2=$(sbatch --dependency=afterok:$JOB1 "$SCRIPT_DIR/run_step2.sh" | grep -oP '\d+')
if [ -z "$JOB2" ]; then
    echo "❌ Step 2 提交失败！"
    exit 1
fi
echo "✅ Step 2 已提交，JOBID: $JOB2 (等待 JOB $JOB1 完成)"

# Step 3: 提交第三个任务，依赖于 Step 2
echo "📤 提交 Step 3 (run_step3.sh)，依赖于 JOB $JOB2..."
JOB3=$(sbatch --dependency=afterok:$JOB2 "$SCRIPT_DIR/run_step3.sh" | grep -oP '\d+')
if [ -z "$JOB3" ]; then
    echo "❌ Step 3 提交失败！"
    exit 1
fi
echo "✅ Step 3 已提交，JOBID: $JOB3 (等待 JOB $JOB2 完成)"

echo ""
echo "=========================================="
echo "✅ 所有任务已提交！"
echo "=========================================="
echo "JOB ID 列表："
echo "  Step 1: $JOB1"
echo "  Step 2: $JOB2 (依赖: $JOB1)"
echo "  Step 3: $JOB3 (依赖: $JOB2)"
echo ""
echo "查看任务状态:"
echo "  squeue -u \$USER"
echo "  squeue -j $JOB1,$JOB2,$JOB3"
echo ""
echo "取消所有任务:"
echo "  scancel $JOB1 $JOB2 $JOB3"
echo "=========================================="


