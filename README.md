# Entity recognition using Deep Learning

Please visit my medium link to see the explanation of the project.

```
with open(self.filename) as f:
    words, tags = [], []
    for line in f:
        line = line.strip()
        if (len(line) == 0 or line.startswith("-DOCSTART-")):
            if len(words) != 0:
                niter += 1
                if self.max_iter is not None and niter > self.max_iter:
                    break
                yield words, tags
                words, tags = [], []
        else:
            ls = line.split(' ')
            word, tag = ls[0],ls[-1]
            if self.processing_word is not None:
                word = self.processing_word(word)
            if self.processing_tag is not None:
                tag = self.processing_tag(tag)
            words += [word]
            tags += [tag]
```
