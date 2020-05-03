from cgan.trainer import Trainer
from cgan.preprocessor import Preprocessor


train_dataset = Preprocessor().get_preprocessed_train_data()
Trainer().train(train_dataset)