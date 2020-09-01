from glob import glob
from absl import app, flags, logging
from tqdm import tqdm
import subprocess

FLAGS = flags.FLAGS

flags.DEFINE_string('dir', default=None,
                    help='Path of dataset to transform')


def main(argv):
    file_list = glob(FLAGS.dir + '/*.hwp')
    logging.info(f'Transforming {len(file_list)} hwp files.')
    for file in tqdm(file_list, desc='transforming'):
        with open(file[:-3] + 'txt', 'w') as f:
            subprocess.call(['hwp5txt', file], stdout=f)


if __name__ == '__main__':
    flags.mark_flags_as_required(['dir'])
    app.run(main)