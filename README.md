# Entity recognition using Deep Learning

Please visit my medium link to see the explanation of the project.

```
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

```
