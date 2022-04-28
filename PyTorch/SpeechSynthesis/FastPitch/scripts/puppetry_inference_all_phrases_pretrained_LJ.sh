#!/usr/bin/env bash

#TODO: update for including spectral tilt and using surface tilt model
#TODO: automate process for whatever inference files are available
#TODO: make more parallel
#TODO: then 3 more times, for surface+source model, source model, no tilt model again
#for FILE in phrases/puppetry/LJ_test_set/*
#do
#  echo $FILE
#  WAV=$(basename $FILE .txt)
#  REF_WAVS_LOC='/home/emelie/Repos/FastPitches/PyTorch/SpeechSynthesis/FastPitch/LJSpeech-1.1/wavs/'
#  OUTPUT='/media/emelie/EMELIESSD/puppeteering/'
#  # copy reference recording
#  cp ${REF_WAVS_LOC}${WAV}.wav ${OUTPUT}${WAV}_recording.wav
#  # synthesise and save as such
#  # PRETRAINED MODEL
#  # 1st argument: phrase, 2nd argument: outputdir,  3rd argument: ref wav
#  ./scripts/puppetry_inference/pretrained_LJ/inference_puppetry_no_reference.sh $FILE $OUTPUT ${REF_WAVS_LOC}${WAV}.wav
#  cp ${OUTPUT}audio_0.wav ${OUTPUT}${WAV}_synth_no_puppet.wav
#
#  # synthesise after puppeteering pitch and save as such
#  ./scripts/puppetry_inference/pretrained_LJ/inference_puppetry_pitch.sh $FILE $OUTPUT ${REF_WAVS_LOC}${WAV}.wav
#  cp ${OUTPUT}audio_0.wav ${OUTPUT}${WAV}_synth_pitch.wav
#
#  # synthesise after puppeteering dur and save as such
#
#  # synthesise after puppeteering energy and save as such
#
#  # synthesise after puppeteering pitch & dur and save as such
#  ./scripts/puppetry_inference/pretrained_LJ/inference_puppetry_pitch_dur.sh $FILE $OUTPUT ${REF_WAVS_LOC}${WAV}.wav
#  cp ${OUTPUT}audio_0.wav ${OUTPUT}${WAV}_synth_pitch_dur.wav
#
#  # synthesise after puppeteering pitch & energy and save as such
#
#  # synthesise after puppeteering dur & energy and save as such
#
#  # synthesise after puppeteering pitch, dur, and energy and save as such
#  ./scripts/puppetry_inference/pretrained_LJ/inference_puppetry_pitch_energy_dur.sh $FILE $OUTPUT ${REF_WAVS_LOC}${WAV}.wav
#  cp ${OUTPUT}audio_0.wav ${OUTPUT}${WAV}_puppet_pitch_energy_dur.wav
#
#done

#TODO add oh my god section


#TODO CHANGE FILENAMING CONVENTION to use 60 phrases only
for WAV_FILE in /media/emelie/EMELIESSD/emphasis_recordings_downsampled22/*
do
  echo $WAV_FILE
  WAV_PREFIX=$(basename $WAV_FILE .wav)
  TXT=${WAV_PREFIX/_[OVSQ]/}.txt
  REF_WAVS_LOC='/media/emelie/EMELIESSD/emphasis_recordings_downsampled22/'
  PHRASES_LOC='./phrases/puppetry/emphasis/'
  OUTPUT='/media/emelie/EMELIESSD/puppeteering/pretrained_LJ/'

  # copy reference recording
  # cp ${WAV_FILE} ${OUTPUT}${WAV_PREFIX}_recording.wav

  # synthesise and save as such
  # 1st argument: phrase, 2nd argument: outputdir,  3rd argument: ref wav
  # ./scripts/puppetry_inference/pretrained_LJ/inference_puppetry_no_reference.sh $PHRASES_LOC$TXT $OUTPUT ${WAV_FILE}
  # cp ${OUTPUT}audio_0.wav ${OUTPUT}${WAV_PREFIX}_synth_no_puppet.wav

  # synthesise after puppeteering pitch and save as such
  # ./scripts/puppetry_inference/pretrained_LJ/inference_puppetry_pitch.sh $PHRASES_LOC$TXT $OUTPUT ${WAV_FILE}
  # cp ${OUTPUT}audio_0.wav ${OUTPUT}${WAV_PREFIX}_synth_pitch.wav

  # synthesise after puppeteering dur and save as such

  # synthesise after puppeteering energy and save as such

  # synthesise after puppeteering pitch & dur and save as such
  # ./scripts/puppetry_inference/pretrained_LJ/inference_puppetry_pitch_dur.sh $PHRASES_LOC$TXT $OUTPUT ${WAV_FILE}
  # cp ${OUTPUT}audio_0.wav ${OUTPUT}${WAV_PREFIX}_synth_pitch_dur.wav

  # synthesise after puppeteering pitch & energy and save as such
  ./scripts/puppetry_inference/pretrained_LJ/inference_puppetry_pitch_energy.sh $PHRASES_LOC$TXT $OUTPUT ${WAV_FILE}
  cp ${OUTPUT}audio_0.wav ${OUTPUT}${WAV_PREFIX}_synth_pitch_energy.wav

  # synthesise after puppeteering dur & energy and save as such

  # synthesise after puppeteering pitch, dur, and energy and save as such
  # ./scripts/puppetry_inference/pretrained_LJ/inference_puppetry_pitch_energy_dur.sh $PHRASES_LOC$TXT $OUTPUT ${WAV_FILE}
  # cp ${OUTPUT}audio_0.wav ${OUTPUT}${WAV_PREFIX}_synth_pitch_energy_dur.wav
done
