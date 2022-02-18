#!/usr/bin/env bash

set -e

: ${DATA_DIR:=LJSpeech-1.1}
: ${ALIGNMENT_DIR:=${DATA_DIR}/mfa_alignments}
: ${ARGS="--extract-mels"}

mfa model download acoustic english
mfa model download dictionary english
mfa validate $DATA_DIR english english
mfa align $DATA_DIR english english $ALIGNMENT_DIR

python prepare_dataset.py \
    --wav-text-filelists filelists/ljs_audio_text.txt \
    --n-workers 16 \
    --batch-size 1 \
    --dataset-path $DATA_DIR \
    --textgrid-path $ALIGNMENT_DIR \
    --extract-pitch \
    --extract-durations\
    --f0-method pyin \
    $ARGS
