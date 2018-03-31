# Entity recognition using Deep Learning

Please visit my medium link to see the explanation of the project.

```
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
```
