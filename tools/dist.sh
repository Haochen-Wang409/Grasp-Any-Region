#!/usr/bin/env bash

set -x

export MASTER_ADDR=${ARNOLD_WORKER_0_HOST}

export PORT=(${ARNOLD_WORKER_0_PORT//,/ })
export NPROC_PER_NODE=${ARNOLD_WORKER_GPU}
export NNODES=${ARNOLD_WORKER_NUM}
export NODE_RANK=${ARNOLD_ID}

FILE=$1
CONFIG=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-$((55500 + $RANDOM % 2000))}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
DEEPSPEED=${DEEPSPEED:-deepspeed_zero2}


if command -v torchrun &> /dev/null
then
  echo "Using torchrun mode."
  TORCHELASTIC_TIMEOUT=18000 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    torchrun --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${PORT} \
    --nproc_per_node=${GPUS} \
    tools/${FILE}.py ${CONFIG} --launcher pytorch --deepspeed $DEEPSPEED "${@:4}"
else
  echo "Using launch mode."
  TORCHELASTIC_TIMEOUT=18000 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m torch.distributed.launch \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${PORT} \
    --nproc_per_node=${GPUS} \
    tools/${FILE}.py ${CONFIG} --launcher pytorch --deepspeed $DEEPSPEED "${@:4}"
fi