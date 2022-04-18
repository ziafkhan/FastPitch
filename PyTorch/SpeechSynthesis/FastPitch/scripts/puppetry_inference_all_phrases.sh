#!/usr/bin/env bash

#TODO: update for including spectral tilt and using surface tilt model
#TODO: then 3 more times, for surface+source model, source model, no tilt model again
for FILE in phrases/puppetry/*
do
  echo $FILE
  WAV=$(basename $FILE .txt)
  REF_WAVS_LOC='/home/emelie/Repos/FastPitches/PyTorch/SpeechSynthesis/FastPitch/LJSpeech-1.1/wavs/'
  OUTPUT='./output/puppeteering/'
  # copy reference recording
  cp ${REF_WAVS_LOC}${WAV}.wav ${OUTPUT}${WAV}_recording.wav
  # synthesise and save as such
  # 1st argument: phrase, 2nd argument: outputdir,  3rd argument: ref wav
  ./scripts/inference_puppetry_no_reference.sh $FILE $OUTPUT ${REF_WAVS_LOC}${WAV}.wav
  cp ${OUTPUT}audio_0.wav ${OUTPUT}${WAV}_synth_no_puppet.wav

  # synthesise after puppeteering pitch and save as such

  # synthesise after puppeteering dur and save as such

  # synthesise after puppeteering energy and save as such

  # synthesise after puppeteering pitch & dur and save as such

  # synthesise after puppeteering pitch & energy and save as such

  # synthesise after puppeteering dur & energy and save as such

  # synthesise after puppeteering pitch, dur, and energy and save as such
  ./scripts/inference_puppetry_pitch_energy_dur.sh $FILE $OUTPUT ${REF_WAVS_LOC}${WAV}.wav
  cp ${OUTPUT}audio_0.wav ${OUTPUT}${WAV}_puppet_pitch.wav

done

#TODO add oh my god section


#TODO add SVO emphasis section
