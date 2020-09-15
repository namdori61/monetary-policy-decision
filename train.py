import json
from configparser import ConfigParser

from absl import app, flags, logging
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger

from preprocess import KbAlbertCharTokenizer
from models import KbAlbertClassificationModel

from knockknock import telegram_sender

FLAGS = flags.FLAGS

flags.DEFINE_string('train_path', default=None,
                    help='Path to the train dataset')
flags.DEFINE_string('dev_path', default=None,
                    help='Path to the dev dataset')
flags.DEFINE_string('test_path', default=None,
                    help='Path to the test dataset')
flags.DEFINE_string('label_type', default=None,
                    help='Label type to train')
flags.DEFINE_string('tokenizer_config_path', default=None,
                    help='Pretrained tokenizer config path')
flags.DEFINE_string('vocab_path', default=None,
                    help='Pretrained tokenizer vocab path')
flags.DEFINE_string('model_path', default=None,
                    help='Pretrained model path')
flags.DEFINE_string('model_config_path', default=None,
                    help='Pretrained model config path')
flags.DEFINE_string('save_dir', default=None,
                    help='Path to save model')
flags.DEFINE_string('version', default=None,
                    help='Explain experiment version')
flags.DEFINE_integer('cuda_device', default=0,
                     help='If given, uses this CUDA device in training')
flags.DEFINE_integer('max_epochs', default=10,
                     help='If given, uses this max epochs in training')
flags.DEFINE_integer('batch_size', default=4,
                     help='If given, uses this batch size in training')
flags.DEFINE_integer('num_workers', default=0,
                     help='If given, uses this number of workers in data loading')
flags.DEFINE_float('lr', default=2e-5,
                   help='If given, uses this learning rate in training')
flags.DEFINE_float('weight_decay', default=0.1,
                   help='If given, uses this weight decay in training')
flags.DEFINE_integer('warm_up', default=500,
                     help='If given, uses this warm up in training')
flags.DEFINE_string('config_path', default=None,
                    help='Path to the config file')


def main(argv):
    f = open(FLAGS.tokenizer_config_path, encoding='UTF-8')
    tokenizer_config = json.loads(f.read())
    tokenizer = KbAlbertCharTokenizer(vocab_file=FLAGS.vocab_path,
                                      pretrained_init_configuration=tokenizer_config)
    if FLAGS.label_type == 'major':
        model = KbAlbertClassificationModel(train_path=FLAGS.train_path,
                                            dev_path=FLAGS.dev_path,
                                            test_path=FLAGS.test_path,
                                            model_path=FLAGS.model_path,
                                            config_path=FLAGS.model_config_path,
                                            tokenizer=tokenizer,
                                            num_classes=3,
                                            batch_size=FLAGS.batch_size,
                                            num_workers=FLAGS.num_workers,
                                            lr=FLAGS.lr,
                                            weight_decay=FLAGS.weight_decay,
                                            warm_up=FLAGS.warm_up)
    elif FLAGS.label_type == 'minor':
        model = KbAlbertClassificationModel(train_path=FLAGS.train_path,
                                            dev_path=FLAGS.dev_path,
                                            test_path=FLAGS.test_path,
                                            model_path=FLAGS.model_path,
                                            config_path=FLAGS.model_config_path,
                                            tokenizer=tokenizer,
                                            num_classes=4,
                                            batch_size=FLAGS.batch_size,
                                            num_workers=FLAGS.num_workers,
                                            lr=FLAGS.lr,
                                            weight_decay=FLAGS.weight_decay,
                                            warm_up=FLAGS.warm_up)
    else:
        ValueError('Unknown model type')

    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(
        filepath=FLAGS.save_dir + '/' + FLAGS.version,
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=2,
        strict=False,
        verbose=False,
        mode='min'
    )

    logger = TensorBoardLogger(
        save_dir=FLAGS.save_dir,
        name='logs_' + FLAGS.label_type,
        version=FLAGS.version
    )
    lr_logger = LearningRateLogger()

    if FLAGS.cuda_device > 1:
        trainer = Trainer(deterministic=True,
                          gpus=FLAGS.cuda_device,
                          distributed_backend='ddp',
                          log_gpu_memory=True,
                          checkpoint_callback=checkpoint_callback,
                          check_val_every_n_epoch=1,
                          early_stop_callback=early_stop,
                          max_epochs=FLAGS.max_epochs,
                          logger=logger,
                          callbacks=[lr_logger])
        logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logging.info(f'Use the number of GPU: {FLAGS.cuda_device}')
    elif FLAGS.cuda_device == 1:
        trainer = Trainer(deterministic=True,
                          gpus=FLAGS.cuda_device,
                          log_gpu_memory=True,
                          checkpoint_callback=checkpoint_callback,
                          check_val_every_n_epoch=1,
                          early_stop_callback=early_stop,
                          max_epochs=FLAGS.max_epochs,
                          logger=logger,
                          callbacks=[lr_logger])
        logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logging.info(f'Use the number of GPU: {FLAGS.cuda_device}')
    else:
        trainer = Trainer(deterministic=True,
                          checkpoint_callback=checkpoint_callback,
                          check_val_every_n_epoch=1,
                          early_stop_callback=early_stop,
                          max_epochs=FLAGS.max_epochs,
                          logger=logger,
                          callbacks=[lr_logger])
        logging.info('No GPU available, using the CPU instead.')

    parser = ConfigParser()
    parser.read(FLAGS.config_path)

    @telegram_sender(token=parser.get('telegram', 'token'),
                     chat_id=parser.get('telegram', 'chat_id'))
    def train_notify(model, trainer):
        results = trainer.fit(model)
        return results

    train_notify(model, trainer)

    if FLAGS.label_type == 'major':
        model.text_embedding.save_pretrained(FLAGS.save_dir)

    if FLAGS.test_path:
        @telegram_sender(token=parser.get('telegram', 'token'),
                         chat_id=parser.get('telegram', 'chat_id'))
        def test_notify(trainer):
            results = trainer.test()
            return results

        test_notify(trainer)


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'train_path', 'dev_path', 'label_type', 'vocab_path', 'model_path',
        'model_config_path', 'save_dir', 'version', 'config_path'
    ])
    app.run(main)