#!/bin/bash
set -x

unset NCCL_DEBUG

if [ "$#" -ge 2 ]; then
  MODEL_DIR=$(realpath "$1")
  GPUS=$2
else
  echo "Usage: $0 <local model dir> <gpu number> [--mps]"
  exit 1
fi

if [ "$3" == "--mps" ]; then
  MPS=true
else
  MPS=false
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT" || exit 1

PY_FILE=./generate.py
if $MPS; then
    LAUNCH_CMD="python $PY_FILE"
    LOW_MEM_OPTS="--offload_model true --t5_cpu"
else
    LAUNCH_CMD="torchrun --nproc_per_node=$GPUS $PY_FILE"
    LOW_MEM_OPTS=""
fi


function t2v_A14B() {
    CKPT_DIR="$MODEL_DIR/Wan2.2-T2V-A14B"

    if $MPS; then
        echo -e "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> t2v_A14B MPS Test: "
        $LAUNCH_CMD --task t2v-A14B --ckpt_dir $CKPT_DIR --size 832*480 $LOW_MEM_OPTS
    else
        echo -e "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> t2v_A14B Multiple GPU Test: "
        $LAUNCH_CMD --task t2v-A14B --ckpt_dir $CKPT_DIR --size 832*480 --dit_fsdp --t5_fsdp --ulysses_size $GPUS

        echo -e "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> t2v_A14B Multiple GPU Test: "
        $LAUNCH_CMD --task t2v-A14B --ckpt_dir $CKPT_DIR --size 720*1280 --dit_fsdp --t5_fsdp --ulysses_size $GPUS

        echo -e "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> t2v_A14B Multiple GPU Test: "
        $LAUNCH_CMD --task t2v-A14B --ckpt_dir $CKPT_DIR --size 1280*720 --dit_fsdp --t5_fsdp --ulysses_size $GPUS

        echo -e "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> t2v_A14B Multiple GPU, prompt extend local_qwen: "
        $LAUNCH_CMD --task t2v-A14B --ckpt_dir $CKPT_DIR --size 480*832 --dit_fsdp --t5_fsdp --ulysses_size $GPUS --use_prompt_extend --prompt_extend_model "Qwen/Qwen2.5-3B-Instruct" --prompt_extend_target_lang "en"
    fi
}


function i2v_A14B() {
    CKPT_DIR="$MODEL_DIR/Wan2.2-I2V-A14B"

    if $MPS; then
        echo -e "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> i2v_14B MPS Test: "
        $LAUNCH_CMD --task i2v-A14B --ckpt_dir $CKPT_DIR --size 832*480 $LOW_MEM_OPTS
    else
        echo -e "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> i2v_14B Multiple GPU Test: "
        $LAUNCH_CMD --task i2v-A14B --ckpt_dir $CKPT_DIR --size 832*480 --dit_fsdp --t5_fsdp --ulysses_size $GPUS

        echo -e "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> i2v_14B Multiple GPU, prompt extend local_qwen: "
        $LAUNCH_CMD --task i2v-A14B --ckpt_dir $CKPT_DIR --size 720*1280 --dit_fsdp --t5_fsdp --ulysses_size $GPUS --use_prompt_extend --prompt_extend_model "Qwen/Qwen2.5-VL-3B-Instruct" --prompt_extend_target_lang "en"

        echo -e "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> i2v_14B Multiple GPU, prompt extend local_qwen: "
        $LAUNCH_CMD --task i2v-A14B --ckpt_dir $CKPT_DIR --size 1280*720 --dit_fsdp --t5_fsdp --ulysses_size $GPUS --use_prompt_extend --prompt_extend_model "Qwen/Qwen2.5-VL-3B-Instruct" --prompt_extend_target_lang "en"

        if [ -n "${DASH_API_KEY+x}" ]; then
            echo -e "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> i2v_14B Multiple GPU, prompt extend dashscope: "
            $LAUNCH_CMD --task i2v-A14B --ckpt_dir $CKPT_DIR --size 480*832 --dit_fsdp --t5_fsdp --ulysses_size $GPUS --use_prompt_extend --prompt_extend_method "dashscope"
        else
            echo -e "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> No DASH_API_KEY found, skip the dashscope extend test."
        fi
    fi
}

function ti2v_5B() {
    CKPT_DIR="$MODEL_DIR/Wan2.2-TI2V-5B"

    if $MPS; then
        echo -e "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ti2v_5B MPS Test: "
        $LAUNCH_CMD --task ti2v-5B --ckpt_dir $CKPT_DIR --size 1280*704 $LOW_MEM_OPTS
    else
        echo -e "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ti2v_5B t2v Multiple GPU Test: "
        $LAUNCH_CMD --task ti2v-5B --ckpt_dir $CKPT_DIR --size 1280*704 --dit_fsdp --t5_fsdp --ulysses_size $GPUS

        echo -e "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ti2v_5B t2v Multiple GPU, prompt extend local_qwen: "
        $LAUNCH_CMD --task ti2v-5B --ckpt_dir $CKPT_DIR --size 704*1280 --dit_fsdp --t5_fsdp --ulysses_size $GPUS --use_prompt_extend --prompt_extend_model "Qwen/Qwen2.5-3B-Instruct" --prompt_extend_target_lang "en"

        echo -e "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ti2v_5B i2v Multiple GPU Test: "
        $LAUNCH_CMD --task ti2v-5B --ckpt_dir $CKPT_DIR --size 704*1280 --dit_fsdp --t5_fsdp --ulysses_size $GPUS --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." --image "examples/i2v_input.JPG"

        echo -e "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ti2v_5B i2v Multiple GPU, prompt extend local_qwen: "
        $LAUNCH_CMD --task ti2v-5B --ckpt_dir $CKPT_DIR --size 1280*704 --dit_fsdp --t5_fsdp --ulysses_size $GPUS --use_prompt_extend --prompt_extend_model "Qwen/Qwen2.5-3B-Instruct" --prompt_extend_target_lang 'en' --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." --image "examples/i2v_input.JPG"

    fi
}

t2v_A14B
i2v_A14B
ti2v_5B
