import numpy as np
import os 
import tensorflow as tf
from config import Config
from sklearn.model_selection import train_test_split
from dataobject import CoNLLDataset
from data_utils import UNK, NUM, NONE, load_vocab, get_processing_word, get_trimmed_glove_vectors

class Model():
    
    def __init__(self, config):
        self.config = config
        self.vocab_words = load_vocab(self.config.filename_words)
        self.vocab_tags  = load_vocab(self.config.filename_tags)
        self.vocab_chars = load_vocab(self.config.filename_chars)

        # Get pre-trained embeddings
        self.w_embeddings = (get_trimmed_glove_vectors(config.filename_trimmed) if self.config.use_pretrained else None)

    def build(self):
        self.tf_placeholders()
        self.word_embeddings()
        self.tf_logits()
        self.tf_prediction_no_crf()
        self.tf_loss()
        self.tf_optimiser()
        self.initialize_session() 

    def tf_placeholders(self):

        self.word_ids_tensor = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")

        self.sequence_lengths_tensor = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        self.char_ids_tensor = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids")

        self.word_lengths_tensor = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")

        self.labels_tensor = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        self.dropout_tensor = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr_tensor = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")

    """Defines self.word_embeddings
        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
    def word_embeddings(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(
                        self.w_embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids_tensor, name="word_embeddings")

        with tf.variable_scope("chars"):
            # get char embeddings matrix
            _char_embeddings = tf.get_variable(
                    name="_char_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nchars, self.config.dim_char])
            char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                    self.char_ids_tensor, name="char_embeddings")

            # put the time dimension on axis=1
            s = tf.shape(char_embeddings)
            char_embeddings = tf.reshape(char_embeddings,
                    shape=[s[0]*s[1], s[-2], self.config.dim_char])
            word_lengths = tf.reshape(self.word_lengths_tensor, shape=[s[0]*s[1]])

            # bi lstm on chars
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                    state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                    state_is_tuple=True)
            _output = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, char_embeddings,
                    sequence_length=word_lengths, dtype=tf.float32)

            # read and concat output
            _, ((_, output_fw), (_, output_bw)) = _output
            output = tf.concat([output_fw, output_bw], axis=-1)

            # shape = (batch size, max sentence length, char hidden size)
            output = tf.reshape(output,
                    shape=[s[0], s[1], 2*self.config.hidden_size_char])
            word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout_tensor)

    """
    For each word in each sentence of the batch, it corresponds to a vector
    of scores, of dimension equal to the number of tags.
    """
    def tf_logits(self):

        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths_tensor, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout_tensor)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])

    """Defines self.labels_pred

    This op is defined only in the case where we don't use a CRF since in
    that case we can make the prediction "in the graph" (thanks to tf
    functions in other words). With theCRF, as the inference is coded
    in python and not in pure tensroflow, we have to make the prediciton
    outside the graph.
    """
    def tf_prediction_no_crf(self):
        if not self.config.use_crf:
            self.labels_pred_not_crf = tf.cast(tf.argmax(self.logits, axis=-1),
                    tf.int32)


    """Defines the loss"""
    def tf_loss(self):
        if self.config.use_crf:
            log_likelihood, _trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels_tensor, self.sequence_lengths_tensor)
            self.trans_params = _trans_params # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels_tensor)
            mask = tf.sequence_mask(sequence_lengths_tensor)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)


    """Defines self.train_op that performs an update on a batch
    Args:
        lr_method: (string) sgd method, for example "adam"
        lr: (tf.placeholder) tf.float32, learning rate
        loss: (tensor) tf.float32 loss to minimize
        clip: (python float) clipping of gradient. If < 0, no clipping

    """
    def tf_optimiser(self):
        with tf.variable_scope("train_step"):
            optimizer = tf.train.AdamOptimizer(self.lr_tensor)
            self.train_op = optimizer.minimize(self.loss)

    """Defines self.sess and initialize the variables"""
    def initialize_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)

    def restore_session(self, dir_model):
        self.saver.restore(self.sess, dir_model)


    def minibatches(self, data, minibatch_size):
        """
        Args:
            data: generator of (sentence, tags) tuples
            minibatch_size: (int)

        Yields:
            list of tuples

        """
        #print(data)
        x_batch, y_batch = [], []
        for (x,y) in data:
            if len(x_batch) == minibatch_size:
                yield x_batch, y_batch
                x_batch, y_batch = [], []

            if type(x[0]) == tuple:
                x = zip(*x)
            x_batch += [x]
            y_batch += [y]
        if len(x_batch) != 0:
            yield x_batch, y_batch

    def _pad_sequences(self, sequences, pad_tok, max_length):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with

        Returns:
            a list of list where each sublist has same length
        """
        sequence_padded, sequence_length = [], []

        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
            sequence_padded +=  [seq_]
            sequence_length += [min(len(seq), max_length)]

        return sequence_padded, sequence_length


    def pad_sequences(self, sequences, pad_tok, nlevels=1):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with
            nlevels: "depth" of padding, for the case where we have characters ids

        Returns:
            a list of list where each sublist has same length

        """
        if nlevels == 1:
            max_length = max(map(lambda x : len(x), sequences))
            sequence_padded, sequence_length = self._pad_sequences(sequences,
                                                pad_tok, max_length)

        elif nlevels == 2:
            max_length_word = max([max(map(lambda x: len(x), seq))
                                for seq in sequences])
            sequence_padded, sequence_length = [], []
            for seq in sequences:
                # all words are same length now
                sp, sl = self._pad_sequences(seq, pad_tok, max_length_word)
                sequence_padded += [sp]
                sequence_length += [sl]

            max_length_sentence = max(map(lambda x : len(x), sequences))
            sequence_padded, _ = self._pad_sequences(sequence_padded,
                    [pad_tok]*max_length_word, max_length_sentence)
            sequence_length, _ = self._pad_sequences(sequence_length, 0,
                    max_length_sentence)

        return sequence_padded, sequence_length

    def get_feed_dict(self, words, labels=None, _lr=None, _dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        char_ids1, word_ids1 = zip(*words)
        word_ids1, sequence_lengths1 = self.pad_sequences(word_ids1, 0)
        char_ids1, word_lengths1 = self.pad_sequences(char_ids1, pad_tok=0,
            nlevels=2)

        # build feed dictionary
        feed = {
            self.word_ids_tensor: word_ids1,
            self.sequence_lengths_tensor: sequence_lengths1
        }

        feed[self.char_ids_tensor] = char_ids1
        feed[self.word_lengths_tensor] = word_lengths1

        if labels is not None:
            labels, _ = self.pad_sequences(labels, 0)
            feed[self.labels_tensor] = labels

        if _lr is not None:
            feed[self.lr_tensor] = _lr

        if _dropout is not None:
            feed[self.dropout_tensor] = _dropout

        return feed, sequence_lengths1

    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        nbatches = (len(train) + self.config.batch_size - 1) // self.config.batch_size
        
        # iterate over dataset
        for i, (words, labels) in enumerate(self.minibatches(train, self.config.batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout)

            _, train_loss = self.sess.run(
                    [self.train_op, self.loss], feed_dict=fd)
            print("batch: {}, loss:{}".format(i, train_loss))

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        print(msg)

        return metrics["f1"]

    def get_chunk_type(self, tok, idx_to_tag):
        """Args:
            tok: id of token, ex 4
            idx_to_tag: dictionary {4: "B-PER", ...}

        Returns:
            tuple: "B", "PER" """
        
        if tok in idx_to_tag:
            tag_name = idx_to_tag[tok]
            tag_class = tag_name.split('-')[0]
            tag_type = tag_name.split('-')[-1]
            #print(tag_class)
            #print(tag_type)
        else:
            print(tok)
            #print(idx_to_tag)

        return tag_class, tag_type


    def get_chunks(self, seq, tags):
        """Given a sequence of tags, group entities and their position

        Args:
            seq: [4, 4, 0, 0, ...] sequence of labels
            tags: dict["O"] = 4

        Returns:
            list of (chunk_type, chunk_start, chunk_end)

        Example:
            seq = [4, 5, 0, 3]
            tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
            result = [("PER", 0, 2), ("LOC", 3, 4)]"""
        default = tags[NONE]
        idx_to_tag = {idx: tag for tag, idx in tags.items()}
        chunks = []
        chunk_type, chunk_start = None, None
        for i, tok in enumerate(seq):
            # End of a chunk 1
            if tok == default and chunk_type is not None:
                # Add a chunk.
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None

            # End of a chunk + start of a chunk!
            elif tok != default:
                tok_chunk_class, tok_chunk_type = self.get_chunk_type(tok, idx_to_tag)
                if chunk_type is None:
                    chunk_type, chunk_start = tok_chunk_type, i
                elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok_chunk_type, i
            else:
                pass

        # end condition
        if chunk_type is not None:
            chunk = (chunk_type, chunk_start, len(seq))
            chunks.append(chunk)

        return chunks

    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in self.minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                                sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(self.get_chunks(lab, self.vocab_tags))
                lab_pred_chunks = set(self.get_chunks(lab_pred,
                                                    self.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100*acc, "f1": 100*f1}

    def evaluate(self, test):
        """Evaluate model on test set

        Args:
            test: instance of class Dataset

        """
        metrics = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])

    def train(self, train, dev):
        """Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of (sentences, tags)
            dev: dataset

        """
        best_score = 0
        nepoch_no_imprv = 0 # for early stopping
        with tf.Session() as session:
            self.sess.run(tf.global_variables_initializer())
            for epoch in range(self.config.nepochs):
                print("Epoch {:} out of {:}".format(epoch + 1,self.config.nepochs))
                score = self.run_epoch(train, dev, epoch)
                #lr *= lr_decay # decay learning rate

                # early stopping and saving best parameters
                if score >= best_score:
                    nepoch_no_imprv = 0
                    self.save_session()
                    best_score = score
                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= nepoch_no_imprv:
                        break


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, _dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            _logits, _trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(_logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, _trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths
        else:
            labels_pred = self.sess.run(self.labels_pred_not_crf, feed_dict=fd)
            return labels_pred, sequence_lengths

    def processing_word_predict(self, vocab_words, vocab_chars, lowercase, chars, allow_unk, word):
        """Return lambda function that transform a word (string) into list,
        or tuple of (list, id) of int corresponding to the ids of the word and
        its corresponding characters.

        Args:
            vocab: dict[word] = idx

        Returns:
            f("cat") = ([12, 4, 32], 12345)
                    = (list of char ids, word id)

        """
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    def predict(self, words_raw):
        
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """

        idx_to_tag = {idx: tag for tag, idx in self.vocab_tags.items()}
        #print(idx_to_tag)
        words = [self.processing_word_predict(self.vocab_words, self.vocab_chars, True, True, True, w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds

