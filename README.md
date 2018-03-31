# Entity recognition using Deep Learning

Please visit my medium link to see the explanation of the project.

```
 def save_session(self):
    """Saves session = weights"""
    if not os.path.exists(self.config.dir_model):
        os.makedirs(self.config.dir_model)
    self.saver.save(self.sess, self.config.dir_model)

def restore_session(self, dir_model):
    self.saver.restore(self.sess, dir_model)
```
