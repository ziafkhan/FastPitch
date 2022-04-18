#!/usr/bin/env bash

: ${WAVEGLOW:="pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt"}
: ${FASTPITCH:="/home/emelie/Repos/FastPitches/PyTorch/SpeechSynthesis/FastPitch/pretrained_models/fastpitch/nvidia_fastpitch_210824.pt"}
: ${BATCH_SIZE:=32}
: ${PHRASES:="phrases/$1"}
: ${OUTPUT_DIR:="$2"}
: ${LOG_FILE:="$OUTPUT_DIR/nvlog_infer.json"}
: ${AMP:=false}
: ${TORCHSCRIPT:=false}
: ${PHONE:=true}
: ${ENERGY:=true}
: ${DENOISING:=0.01}
: ${WARMUP:=0}
: ${REPEATS:=1}
: ${CPU:=false}

: ${SPEAKER:=1}
: ${NUM_SPEAKERS:=1}

# puppeteering
: ${PUPPET_PITCH:=true}
: ${PUPPET_ENERGY:=true}
: ${PUPPET_DURS:=true}
: ${REF_WAV:="$3"}

echo -e "\nAMP=$AMP, batch_size=$BATCH_SIZE\n"

ARGS=""
ARGS+=" -i $PHRASES"
ARGS+=" -o $OUTPUT_DIR"
ARGS+=" --log-file $LOG_FILE"
ARGS+=" --fastpitch $FASTPITCH"
ARGS+=" --waveglow $WAVEGLOW"
ARGS+=" --wn-channels 256"
ARGS+=" --batch-size $BATCH_SIZE"
ARGS+=" --denoising-strength $DENOISING"
ARGS+=" --repeats $REPEATS"
ARGS+=" --warmup-steps $WARMUP"
ARGS+=" --speaker $SPEAKER"
ARGS+=" --n-speakers $NUM_SPEAKERS"
ARGS+=" --save-mels"
[ "$CPU" = false ]          && ARGS+=" --cuda"
[ "$CPU" = false ]          && ARGS+=" --cudnn-benchmark"
[ "$AMP" = true ]           && ARGS+=" --amp"
[ "$PHONE" = "true" ]       && ARGS+=" --p-arpabet 1.0"
[ "$ENERGY" = "true" ]      && ARGS+=" --energy-conditioning"
[ "$TORCHSCRIPT" = "true" ] && ARGS+=" --torchscript"
[ "$REF_WAV" != "" ]        && ARGS+=" --ref-wav $REF_WAV"
[ "$PUPPET_PITCH" = "true" ] && ARGS+=" --puppet-pitch"
[ "$PUPPET_ENERGY" = "true" ] && ARGS+=" --puppet-energy"
[ "$PUPPET_DURS" = "true" ] && ARGS+="  --puppet-durs"

mkdir -p "$OUTPUT_DIR"

python inference.py $ARGS "$@"
