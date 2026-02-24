#!/bin/bash -l
#SBATCH --job-name=qwen_region_full
#SBATCH --partition=h24gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/qwen_region_full_%A_%a.out
#SBATCH --account=OD-237777
#SBATCH --gres=gpu:4

set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

conda activate deepfake
conda activate vllm

if [[ -n "${SCRIPT_DIR_OVERRIDE:-}" ]]; then
  SCRIPT_DIR="${SCRIPT_DIR_OVERRIDE}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_qwen_region_full.sbatch" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
elif [[ -n "${SLURM_JOB_SCRIPT:-}" && -f "${SLURM_JOB_SCRIPT}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "$(realpath "${SLURM_JOB_SCRIPT}")")" && pwd)"
elif [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
else
  SCRIPT_DIR="$(pwd)"
fi

mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"

MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/LLM/Qwen3-30B-A3B-Instruct-2507/}"
INPUT_MODEL_FOLDER="${INPUT_MODEL_FOLDER:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final__En/Qwen3-VL-30B-A3B-Thinking_}"
INPUT_JSONL="${INPUT_JSONL:-}"
GT_CSV="${GT_CSV:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/final_mask_topk/region_phone_table_topk3.csv}"
SYSTEM_FILE="${SYSTEM_FILE:-${SCRIPT_DIR}/polisher_prompt/region_forensics_system.txt}"
USER_TEMPLATE_FILE="${USER_TEMPLATE_FILE:-${SCRIPT_DIR}/polisher_prompt/region_forensics_user.txt}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/En/polisher/}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
if [[ "${SHARD_COUNT}" -gt 1 ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE_DIR}/shard_${SHARD_ID}_of_${SHARD_COUNT}}"
  OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_BASE_DIR}/qwen_region_outputs_shard_${SHARD_ID}_of_${SHARD_COUNT}.jsonl}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE_DIR}}"
  OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR}/qwen_region_outputs.jsonl}"
fi
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-600}"
DTYPE="${DTYPE:-auto}"
BACKEND="${BACKEND:-vllm}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-20}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"

echo "[run] SCRIPT_DIR=${SCRIPT_DIR}"
echo "[run] MODEL_ID=${MODEL_ID}"
if [[ -z "${INPUT_JSONL:-}" ]]; then
  INPUT_JSONL="${INPUT_MODEL_FOLDER%/}/qwen_region_outputs_shard_0_of_4.jsonl"
fi
echo "[run] INPUT_MODEL_FOLDER=${INPUT_MODEL_FOLDER}"
echo "[run] INPUT_JSONL=${INPUT_JSONL}"
echo "[run] GT_CSV=${GT_CSV}"
echo "[run] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[run] SHARD_ID=${SHARD_ID} SHARD_COUNT=${SHARD_COUNT}"
echo "[run] BATCH_SIZE=${BATCH_SIZE}"
echo "[run] BACKEND=${BACKEND} TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"

EXTRA_ARGS=()
if [[ "${DO_SAMPLE}" == "1" ]]; then
  EXTRA_ARGS+=(--do-sample)
fi
if [[ "${VLLM_ENFORCE_EAGER}" == "1" ]]; then
  EXTRA_ARGS+=(--vllm-enforce-eager)
fi

python "${SCRIPT_DIR}/artifact_polish_prompt.py" \
  --backend "${BACKEND}" \
  --model-id "${MODEL_ID}" \
  --input-jsonl "${INPUT_JSONL}" \
  --gt-csv "${GT_CSV}" \
  --system-file "${SYSTEM_FILE}" \
  --user-template-file "${USER_TEMPLATE_FILE}" \
  --dtype "${DTYPE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --batch-size "${BATCH_SIZE}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --num-shards "${SHARD_COUNT}" \
  --shard-id "${SHARD_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  "${EXTRA_ARGS[@]}" \
  "$@"


