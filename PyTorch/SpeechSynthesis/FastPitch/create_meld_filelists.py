import csv
import datetime
import glob
import json
import math
import os
import pathlib
from functools import lru_cache

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

MELD_LOC = '/media/emelie/EMELIESSD/MELD/MELD.Raw/'
MELD_PROCESSED_LOC = '/media/emelie/EMELIESSD/MELD/MELD_wav22k/'
LJ_PROCESSED_LOC = '/home/emelie/Repos/FastPitches/PyTorch/SpeechSynthesis/FastPitch/LJSpeech-1.1/'
OUTPUT = 'meld_filelists/all_files.txt'
MAIN_CHARACTERS = ['chandler', 'joey', 'rachel', 'monica', 'phoebe', 'ross']
FILTER_MAIN = True
MIX_LJ = True

corrupt_files = ['/media/emelie/EMELIESSD/MELD/MELD.Raw/train/train_splits/dia125_utt3.mp4',
                 '/media/emelie/EMELIESSD/MELD/MELD.Raw/dev/dev_splits_complete/dia110_utt7.mp4',
                 '/media/emelie/EMELIESSD/MELD/MELD_wav22k/wavs/train_Chandler_dia8_utt6.wav',
                 '/media/emelie/EMELIESSD/MELD/MELD_wav22k/wavs/train_The Interviewer_dia0_utt0.wav',
                 '/media/emelie/EMELIESSD/MELD/MELD_wav22k/wavs/train_Chandler_dia0_utt1.wav',
                 '/media/emelie/EMELIESSD/MELD/MELD_wav22k/wavs/train_The Interviewer_dia0_utt2.wav',
                 '/media/emelie/EMELIESSD/MELD/MELD_wav22k/wavs/train_Chandler_dia0_utt3.wav',
                 '/media/emelie/EMELIESSD/MELD/MELD_wav22k/wavs/train_The Interviewer_dia0_utt4.wav',
                 '/media/emelie/EMELIESSD/MELD/MELD_wav22k/wavs/train_Chandler_dia0_utt5.wav',
                 '/media/emelie/EMELIESSD/MELD/MELD_wav22k/wavs/train_The Interviewer_dia0_utt6.wav',
                 '/media/emelie/EMELIESSD/MELD/MELD_wav22k/wavs/train_Chandler_dia0_utt7.wav',
                 '/media/emelie/EMELIESSD/MELD/MELD_wav22k/wavs/train_The Interviewer_dia0_utt8.wav',
                 '/media/emelie/EMELIESSD/MELD/MELD_wav22k/wavs/train_Chandler_dia0_utt9.wav',
                 '/media/emelie/EMELIESSD/MELD/MELD_wav22k/wavs/train_The Interviewer_dia0_utt10.wav',
                 '/media/emelie/EMELIESSD/MELD/MELD_wav22k/wavs/train_Chandler_dia0_utt11.wav',
                 '/media/emelie/EMELIESSD/MELD/MELD_wav22k/wavs/train_The Interviewer_dia0_utt12.wav',
                 '/media/emelie/EMELIESSD/MELD/MELD_wav22k/wavs/train_Chandler_dia0_utt13.wav']


def get_LJ_filelist(split):
    lj_filenames = {
        'train': 'ljs_audio_text_train_v3.txt',
        'test': 'ljs_audio_text_test.txt',
        'dev': 'ljs_audio_text_val.txt',
        'val': 'ljs_audio_test_val.txt'
    }
    if split not in lj_filenames:
        raise FileNotFoundError('No idea what LJ file to use')
    with open(os.path.join('filelists', lj_filenames[split])) as f:
        return f.readlines()


def mix_with_LJ(output_lines, split, speaker_int):
    LJ_lines = get_LJ_filelist(split)
    # how much LJ?
    lj_num = min([len(LJ_lines), math.ceil(0.33 * len(output_lines))])
    lj_with_speaker_int = list(map(lambda line: LJ_PROCESSED_LOC + line.strip() + f'|{speaker_int}',
                                   LJ_lines))
    output_lines.extend(lj_with_speaker_int[:lj_num])
    return output_lines


def length_allowed(metadata_row):
    start_time = datetime.datetime.strptime(metadata_row["StartTime"], '%H:%M:%S,%f')
    end_time = datetime.datetime.strptime(metadata_row["EndTime"], '%H:%M:%S,%f')
    diff = end_time - start_time
    return 1 < diff.total_seconds() <= 10


def get_metadata(split, dialogue, utterance):
    annotation = os.path.join(MELD_LOC, f'utf8_{split}_sent_emo.csv')
    with open(annotation, newline='', encoding='utf8') as f:
        csv_file = csv.DictReader(f, delimiter=',', quotechar='"')
        for row in csv_file:
            if (row["Dialogue_ID"] == dialogue
                    and row["Utterance_ID"] == utterance):
                return row
    raise LookupError('no corresponding metadata found')


def test_fragment(wav_filename_stem):
    split, _, dialogue, utterance = wav_filename_stem.split('_')
    meta = get_metadata(split, dialogue[3:],
                        utterance[3:])  # remove 'dia' and 'utt'
    return length_allowed(meta)


def is_LJ(wav_file):
    return 'LJSpeech' in wav_file


def filelist_to_audio_pitch_text_filelist(in_file, out_file):
    lines = ''
    with open(in_file) as f:
        for line in f:
            wav_file, transcript, speaker_int = line.split('|')
            wav_filename = pathlib.Path(wav_file).stem
            if is_LJ(wav_file):
                loc = LJ_PROCESSED_LOC
            else:
                loc = MELD_PROCESSED_LOC
                if not test_fragment(wav_filename):
                    continue
            new_line = '|'.join([os.path.join(loc, 'wavs', wav_filename + '.wav'),
                                 os.path.join(loc, 'pitch', wav_filename + '.pt'),
                                 transcript,
                                 speaker_int
                                 ])
            lines += new_line
    with open(out_file, 'w') as g:
        g.write(lines)


def save_mp4_as_wav(input_name, output_name):
    if output_name in corrupt_files:
        pass
    audio = AudioSegment.from_file(input_name, 'mp4')
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)
    audio.export(output_name, format='wav')


all_output_lines = list()

known_speakers = {}


@lru_cache()
def speaker_to_int(speaker_key):
    if '-' in speaker_key:
        speaker_key = speaker_key.split('-')[0]
    speaker_int = known_speakers.get(speaker_key, -1)
    if not speaker_int:
        speaker_int = len(known_speakers) + 1
        known_speakers[speaker_key] = speaker_int
    return speaker_int


for split in ['train', 'dev', 'test']:
    # assumes there is only one subfolder per split
    folder = glob.glob(os.path.join(MELD_LOC, split, '*'))[0]
    annotation = os.path.join(MELD_LOC, f'utf8_{split}_sent_emo.csv')
    split_output_lines = list()
    too_short_or_long = set()
    with open(annotation, newline='', encoding='utf8') as f:
        csv_file = csv.DictReader(f, delimiter=',', quotechar='"')
        for row in csv_file:
            start_time = row["StartTime"]
            end_time = row["EndTime"]
            mp4_name = f'dia{row["Dialogue_ID"]}_utt{row["Utterance_ID"]}'
            mp4_path = os.path.join(folder, mp4_name + '.mp4')
            if mp4_path in corrupt_files:
                continue
            speaker = row["Speaker"]
            if '/' in speaker:
                speaker = speaker.replace('/', '-')
            if FILTER_MAIN and speaker.lower() not in MAIN_CHARACTERS:
                continue
            speaker_number = speaker_to_int(speaker)
            wav_name = f'{split}_{speaker}_{mp4_name}'
            if not test_fragment(wav_name):
                too_short_or_long.add(wav_name)
            wav_path = os.path.join(MELD_PROCESSED_LOC, 'wavs', wav_name + '.wav')
            try:
                # save_mp4_as_wav(mp4_path, wav_path)
                pass
            except FileNotFoundError:
                print('File not found', mp4_path)
                continue
            except CouldntDecodeError as e:
                print('Decode issue', mp4_path)
                continue
            annotation = row['Utterance']
            annotation = annotation.replace('...', '')\
                .replace('—', ',')\
                .replace('…', '')
            split_output_lines.append('|'.join([wav_path, annotation, str(speaker_number)]))

    if MIX_LJ:
        split_output_lines = mix_with_LJ(split_output_lines, split,
                                         speaker_int=speaker_to_int('originalLJ'))
    filter_condition = 'all' if not FILTER_MAIN else 'main'
    with open(f'filelists/meld/{split}_{filter_condition}_files.txt', 'w') as f:
        f.writelines('\n'.join(split_output_lines))

    with open(f'filelists/meld/{split}_{filter_condition}_rejects_too_short_or_long.txt', 'w') as g:
        g.write('\n'.join(too_short_or_long))

    print(f'{len(too_short_or_long)} utterances skipped for {split}')

    # make filelist for post processing, pre training
    filelist_to_audio_pitch_text_filelist(f'filelists/meld/{split}_{filter_condition}_files.txt',
                                          f'filelists/meld/meld_audio_pitch_{split}_{filter_condition}_files.txt')

    all_output_lines.extend(split_output_lines)


with open(f'filelists/meld/all_{filter_condition}_files.txt', 'w') as f:
    f.writelines('\n'.join(all_output_lines))

with open(f'filelists/meld/{filter_condition}_speakers.json', 'w') as f:
    json.dump(known_speakers, f, indent=4)

