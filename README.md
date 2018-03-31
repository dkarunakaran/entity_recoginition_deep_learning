# Entity recognition using Deep Learning

Please visit my medium link to see the explanation of the project.

```
W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])
                    
b = tf.get_variable("b", shape=[self.config.ntags],
        dtype=tf.float32, initializer=tf.zeros_initializer())

nsteps = tf.shape(output)[1]
output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
pred = tf.matmul(output, W) + b
self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])
```
