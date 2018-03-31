# Entity recognition using Deep Learning

Please visit my medium link to see the explanation of the project.

```
cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
    cell_fw, cell_bw, self.word_embeddings,
    sequence_length=self.sequence_lengths_tensor, dtype=tf.float32)
output = tf.concat([output_fw, output_bw], axis=-1)
output = tf.nn.dropout(output, self.dropout_tensor)
```
