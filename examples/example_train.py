from cgan.trainer import Trainer
from cgan.preprocessor import Preprocessor
from cgan.utils import load_config


config = load_config('config.yaml')

train_dataset = Preprocessor().get_preprocessed_train_data()
Trainer().train(train_dataset, **config)