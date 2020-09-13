import csv
import json

from absl import app, flags, logging
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', default=None,
                    help='Path of input file')
flags.DEFINE_string('output_path', default=None,
                    help='Path of output file')


def main(argv):
    logging.info(f'Labeling {FLAGS.input_path}')
    with open(FLAGS.input_path, 'r') as input_file,\
            open(FLAGS.output_path, 'w') as output_file:
        reader = csv.reader(input_file)
        next(reader, None)
        for line in tqdm(reader, desc='labeling'):
            labeled_data = {'date': line[0].strip(),
                            'text': line[1].strip(),
                            'major_direction': line[2].strip(),
                            'voting': line[3].strip(),
                            'minor_direction': line[4].strip()}
            # labeling major class
            if line[2] == 'fall':
                labeled_data['label_major'] = 0
            elif line[2] == 'freeze':
                labeled_data['label_major'] = 1
            else:
                labeled_data['label_major'] = 2
            # labeling minor class
            if line[4] == 'fall':
                labeled_data['label_minor'] = 0
            elif line[4] == 'freeze':
                labeled_data['label_minor'] = 1
            elif line[4] == 'rise':
                labeled_data['label_minor'] = 2
            else:
                labeled_data['label_minor'] = 3
            output_file.write(json.dumps(labeled_data, ensure_ascii=False))
            output_file.write('\n')


if __name__ == '__main__':
    flags.mark_flags_as_required(['input_path', 'output_path'])
    app.run(main)
