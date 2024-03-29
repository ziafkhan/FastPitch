#!/usr/bin/env bash

# : ${WAVEGLOW:="pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt"}
: ${WAVEGLOW:="SKIP"}
: ${FASTPITCH:="pretrained_models/fastpitch/FastPitch_checkpoint_300.pt"}
: ${BATCH_SIZE:=4}
: ${PHRASES:="phrases/devset10.tsv"}
: ${OUTPUT_DIR:="./output/audio_$(basename ${PHRASES} .tsv)"}
: ${LOG_FILE:="$OUTPUT_DIR/nvlog_infer.json"}
: ${AMP:=false}
: ${TORCHSCRIPT:=false}
: ${PHONE:=true}
: ${ENERGY:=true}
: ${DENOISING:=0.01}
: ${WARMUP:=0}
: ${REPEATS:=1}
: ${CPU:=true}
: ${SAVE_MELS:=true}
: ${VTL_SCALE:=1}
: ${FORMANT_SHIFT:=-100}
: ${PITCH_TRANSFORM:=1}
: ${PITCH_SHIFT:=0}
: ${SPEAKER:=0}
: ${NUM_SPEAKERS:=109}

echo -e "\nAMP=$AMP, batch_size=$BATCH_SIZE\n"

ARGS=""
ARGS+=" -i $PHRASES"
ARGS+=" -o $OUTPUT_DIR"
ARGS+=" --log-file $LOG_FILE"
ARGS+=" --fastpitch $FASTPITCH"
ARGS+=" --waveglow $WAVEGLOW"
#ARGS+=" --wn-channels 256"
ARGS+=" --batch-size $BATCH_SIZE"
ARGS+=" --denoising-strength $DENOISING"
ARGS+=" --repeats $REPEATS"
ARGS+=" --warmup-steps $WARMUP"
ARGS+=" --speaker $SPEAKER"
ARGS+=" --n-speakers $NUM_SPEAKERS"
ARGS+=" --formant-scale $VTL_SCALE"
ARGS+=" --formant-shift $FORMANT_SHIFT"
ARGS+=" --pitch-transform-amplify $PITCH_TRANSFORM"
ARGS+=" --pitch-transform-shift $PITCH_SHIFT"
[ "$CPU" = false ]          && ARGS+=" --cuda"
[ "$CPU" = false ]          && ARGS+=" --cudnn-benchmark"
[ "$AMP" = true ]           && ARGS+=" --amp"
[ "$PHONE" = "true" ]       && ARGS+=" --p-arpabet 1.0"
[ "$ENERGY" = "true" ]      && ARGS+=" --energy-conditioning"
[ "$TORCHSCRIPT" = "true" ] && ARGS+=" --torchscript"
[ "$SAVE_MELS" = "true" ]   && ARGS+=" --save-mels"

mkdir -p "$OUTPUT_DIR"

python inference.py $ARGS "$@"
