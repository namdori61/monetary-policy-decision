import re
from glob import glob

import hanja
from absl import app, flags, logging
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir', default=None,
                    help='Path of input directory to preprocess')
flags.DEFINE_string('output_path', default=None,
                    help='Path of output file')


def main(argv):
    file_list = glob(FLAGS.input_dir + '/*.txt')
    file_list = sorted(file_list)

    logging.info(f'Preprocessing {len(file_list)} txt files to {FLAGS.output_path}')
    with open(FLAGS.output_path, 'w') as output_file:
        text_list = []
        output_file.write('date, text, major_direction, voting, minor_direction \n')
        for file in tqdm(file_list, desc='preprocessing'):
            with open(file, 'r') as f:
                date = file[-14:-4].replace('-', '')
                text = f.read().replace('\n', ' ')

                # hanja translate
                text = text.strip()
                text = ''.join([hanja.translate(c, 'substitution') for c in text])

                # remove special characters
                text = re.sub(pattern='[^\w\s]', repl='', string=text)
                text = re.sub(pattern='\s{1,}', repl=' ', string=text)

                text_list.append('"' + date + '", "' + text + '"')

        # sort by date
        sorted(text_list, key=lambda x: x[:9], reverse=True)
        for text in tqdm(text_list, desc='writing output'):
            output_file.write(text + ' \n')


if __name__ == '__main__':
    flags.mark_flags_as_required(['input_dir', 'output_path'])
    app.run(main)
