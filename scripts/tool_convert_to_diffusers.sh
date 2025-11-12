DIFFUSERS_PATH=${1:-/projects/bdbk/cherd/diffusers}
CHECKPOINT_PATH=${2:-/projects/bdbk/cherd/rendersynth/ControlNet/checkpoint/vectorsynth-clip/vectorsynth-clip.ckpt}
ORIGINAL_CONFIG_FILE=${3:-/projects/bdbk/cherd/rendersynth/scripts/models/cldm_v21.yaml}
DUMP_PATH=${4:-/projects/bdbk/cherd/rendersynth/scripts/models/dump}
DEVICE=${5:-cpu}

python ${DIFFUSERS_PATH}/scripts/convert_original_controlnet_to_diffusers.py \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --original_config_file ${ORIGINAL_CONFIG_FILE} \
    --dump_path ${DUMP_PATH} \
    --device ${DEVICE} \
    --to_safetensor