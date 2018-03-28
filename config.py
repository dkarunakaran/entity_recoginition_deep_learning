class Config():

    # shared global variables to be imported from model also
    UNK = "$UNK$"
    NUM = "$NUM$"
    NONE = "O"

    # dataset
    filename_dev =  "data/coNLL/eng.validate.iob"
    filename_test = filename_train  = "data/coNLL/eng.train.iob"

    max_iter = None # if not None, max number of examples in Dataset


    # vocab (created from dataset with build_data.py)
    filename_words = "data/words.txt"
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"

    # embeddings
    dim_word = 300
    dim_char = 100

    #filename_glove = "data/glove.6B/glove.6B.300d.txt"
    filename_glove = "data/glove.840B/glove.840B.300d.txt"
    # trimmed embeddings (created from glove_filename with build_data.py)
    #filename_trimmed = "data/glove.6B.300d.trimmed.npz"
    filename_trimmed = "data/glove.840B.300d.trimmed.npz"


    use_pretrained = True

    # general config
    dir_output = "results/test/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"
    
    # embeddings
    dim_word = 300
    dim_char = 100

    nwords  = 10
    nchars  = 100#10
    ntags   = 10

    # training
    use_crf = True
    train_embeddings = False
    nepochs          = 40
    dropout          = 0.5
    #batch_size       = 20
    batch_size       = 64
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 5

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

