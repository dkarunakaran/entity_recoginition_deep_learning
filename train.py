from config import Config
from dataobject import CoNLLDataset
from data_utils import get_processing_word
from model import Model

config = Config()
model = Model(config)
model.build()

processing_word = get_processing_word(model.vocab_words, model.vocab_chars, lowercase=True, chars=True)
processing_tag  = get_processing_word(model.vocab_tags,lowercase=False, allow_unk=False)
dev_data   = CoNLLDataset(config.filename_dev, processing_word, processing_tag, config.max_iter)
train_data = CoNLLDataset(config.filename_train, processing_word, processing_tag, config.max_iter)
model.train(train_data, dev_data)