import warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#默认为0：输出所有log信息
#设置为1：进一步屏蔽INFO信息
#设置为2：进一步屏蔽WARNING信息
#设置为3：进一步屏蔽ERROR信息
class Config(object):
    # data_path parameters
    UNK_ID = 1
    SOS_ID = 2
    EOS_ID = 3
    BATCH_SIZE = 64
    LEARNING_RATE = 1.0
    NUM_LAYERS = 2
    SRC_VOCAB_SIZE = 45000
    TGT_VOCAB_SIZE = 28000
    PAR_VOCAB_SIZE = 45000
    KEEP_PROB = 0.7
    MAX_GRAD_NORM = 5
    EPOCHS = 15
    PARA_HIDDEN_SIZE = 100
    HIDDEN_SIZE = 600
    MAX_DEC_LEN = 100
    MAX_LEN_S = 100
    MAX_LEN_Q = 50
    MAX_LEN_P = 150
    REPLACE_UNK = False
    BEAM_SIZE = 3
    # BEAM_SEARCH = True
    MODE = 'train'

    SRC_TRAIN_DATA = "../data2.0/src-train.txt.id"
    TRG_TRAIN_DATA = "../data2.0/tgt-train.txt.id"
    PARA_TRAIN_DATA = "../data2.0/para-train.txt.id"
    SRC_DEV_DATA = "../data2.0/src-dev.txt.id"
    TRG_DEV_DATA = "../data2.0/tgt-dev.txt.id"
    PARA_DEV_DATA = "../data2.0/para-dev.txt.id"
    SRC_TEST_DATA = "../data2.0/src-test.txt.id"
    TRG_TEST_DATA = "../data2.0/tgt-test.txt.id"
    PARA_TEST_DATA = "../data2.0/para-test.txt.id"
    TEST_SENTENCES = "../test_sentences.txt"
    TEST_QUESTIONS= "../test_questions.txt"
    PREDICTED_QUESTIONS = "../predicted_questions."
    SRC_ID2EMBEDDING = "../data2.0/src_embedding_file.npy"
    TGT_ID2EMBEDDING = "../data2.0/tgt_embedding_file.npy"
    PARA_ID2EMBEDDING = "../data2.0/para_embedding_file.npy"
    SRC_ID2WORD = "../data2.0/src_i2w_file"
    SRC_WORD2ID = "../data2.0/src_w2i_file"
    TGT_ID2WORD = "../data2.0/tgt_i2w_file"
    TGT_WORD2ID = "../data2.0/tgt_w2i_file"
    PARA_ID2WORD = "../data2.0/para_i2w_file"
    PARA_WORD2ID = "../data2.0/para_w2i_file"
    CHECKPOINT_PATH = "../checkpoints/qgmodel-ckpt"
    CHECKPOINT_DIR = "../checkpoints/"
    PARA_CHECKPOINT_PATH = "../para_checkpoints/pqgmodel-ckpt"
    PARA_CHECKPOINT_DIR = "../para_checkpoints/"
    SUMMARY_DIR = "../summary/"
    SUMMARY_EVAL_DIR = "../summary_eval/"

def parse(self,kwargs):
    for k,v in kwargs.items():
        if not hasattr(self,k):
            warnings.warn("your config doesn't have that key(%s)" %(k))
        else:
            setattr(self,k,v)
    print('user config:')
    print('#################################')
    for k in dir(self):
        if not k.startswith('_') and k != 'parse' and k != 'state_dict':
            print(k, getattr(self, k))
    print('#################################')
    return self
 
def state_dict(self):
    return {k: getattr(self, k) for k in dir(self) if not k.startswith('_') and k != 'parse' and k != 'state_dict'}
 
Config.parse = parse
Config.state_dict = state_dict
opt = Config()
