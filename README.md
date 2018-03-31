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
            
            
def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False, allow_unk=True):

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    print(word)
                    print(vocab_words)

        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f
```
