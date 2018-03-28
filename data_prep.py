import numpy as np
import os
import tensorflow as tf
from config import Config
from sklearn.model_selection import train_test_split
from dataobject import CoNLLDataset
from data_utils import get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word

# Create instance of config
config = Config()

processing_word = get_processing_word(lowercase=True)

# Generators
dev   = CoNLLDataset(config.filename_dev, processing_word)
test  = CoNLLDataset(config.filename_test, processing_word)
train = CoNLLDataset(config.filename_train, processing_word)

# Build Word and Tag vocab
vocab_words, vocab_tags = get_vocabs([train, dev, test])
vocab_glove = get_glove_vocab(config.filename_glove)
vocab = vocab_words & vocab_glove
vocab.add(config.UNK)
vocab.add(config.NUM)

# Save vocab
write_vocab(vocab, config.filename_words)
write_vocab(vocab_tags, config.filename_tags)

# Trim GloVe Vectors
vocab = load_vocab(config.filename_words)
export_trimmed_glove_vectors(vocab, config.filename_glove,
                            config.filename_trimmed, config.dim_word)

# Build and save char vocab
train = CoNLLDataset(config.filename_train)
vocab_chars = get_char_vocab(train)
write_vocab(vocab_chars, config.filename_chars)

