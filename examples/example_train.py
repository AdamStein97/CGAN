from cgan.src.trainer import Trainer
from cgan.src.preprocessor import Preprocessor


train_dataset = Preprocessor().get_preprocessed_train_data()
Trainer().train(train_dataset)