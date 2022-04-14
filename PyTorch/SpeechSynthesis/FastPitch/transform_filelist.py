import argparse
import os.path
from pathlib import Path

def write_file(filename, info):
    with open(filename) as f:
        f.writelines('\n'.join(info))

def main(input_file, output_file, val_file=None):
    all_info = list()
    with open(USBORNE_FILE) as f:
        for line in f:
            filename, transcript = line.strip().split('|', maxsplit=1)
            name_stem = Path(os.path.basename(filename)).stem
            all_info.append('|'.join([f'mels/{name_stem}.pt',
                                      f'durations/{name_stem}.pt',
                                      f'pitch_char/{name_stem}.pt',
                                      transcript]))
    dir_path = os.path.dirname(input_file)
    if not val_file:
        write_file(os.path.join(dir_path, output_file), all_info)
    else:
        # arbitrary split to try
        train_info = all_info[:2000]
        val_info = all_info[2000:]
        write_file(os.path.join(dir_path, output_file), train_info)
        write_file(os.path.join(dir_path, val_file), val_info)        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, required=True
    parser.add_argument('-v', '--val-file', type=str, required=False, default=None)
    args, _ = parser.parse_args()
    main(args.input_file, args.output_file, args.val_file)
