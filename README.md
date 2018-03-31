# Entity recognition using Deep Learning

Please visit my medium link to see the explanation of the project.

```
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
```
