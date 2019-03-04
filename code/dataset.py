import numpy as np
from tqdm import tqdm
import random

class Batcher(object):
    def __init__(self,params):
        self.params = params
        self.batch_size = self.params.BATCH_SIZE
        if self.params.MODE == 'train':
            self.src = self.params.SRC_TRAIN_DATA#sentence ids eos作为结尾
            self.trg = self.params.TRG_TRAIN_DATA#question ids eos作为结尾
        elif self.params.MODE == 'val':
            self.src = self.params.SRC_DEV_DATA
            self.trg = self.params.TRG_DEV_DATA
        elif self.params.MODE == 'test':
            self.src = self.params.SRC_TEST_DATA
            self.trg = self.params.TRG_TEST_DATA
        with open(self.src,'r') as f:
            self.src_lines = f.readlines()
            self.len = len(self.src_lines)
        with open(self.trg,'r') as f:
            self.trg_lines = f.readlines()
            assert self.len == len(self.trg_lines)
        # self.data = zip(self.src_lines,self.trg_lines)
        
        self.process()
        self.reset()

    def reset(self):
        self.start = 0
        # self.use_up = False
        if self.params.MODE == 'train':
            random.shuffle(self.indexs)

    def process(self):
        self.src_input = []
        self.src_size = []
        self.trg_target = []
        self.trg_size = []
        print("processing %s dataset..."%(self.params.MODE))
        for (src_line,trg_line) in tqdm(zip(self.src_lines,self.trg_lines)):
            src_line = src_line.strip().split()#字符串转为list
            src_line = [int(x) for x in src_line]#转为int        
            trg_line = trg_line.strip().split()#字符串转为list
            trg_line = [int(x) for x in trg_line]#转为int
            if len(src_line)<self.params.MAX_LEN_S and len(src_line)>1\
                and len(trg_line)<self.params.MAX_LEN_Q and len(trg_line)>1:
                self.src_size.append(len(src_line))
                self.src_input.append(src_line)
                self.trg_size.append(len(trg_line))
                self.trg_target.append(trg_line)
        
        self.len = len(self.src_input)
        self.indexs = list(range(self.len))


    def get_batch(self):
        while True:
            if self.start >= self.len:
                self.reset()
                return
                
            self.end = min(self.start + self.batch_size,self.len)
            src_input_batch = []
            src_size_batch = []
            trg_input_batch = []
            trg_target_batch = []
            trg_size_batch = []
            for i in range(self.start,self.end):
                index = self.indexs[i]
                src_size_batch.append(self.src_size[index])
                trg_size_batch.append(self.trg_size[index])

            max_src_size = max(src_size_batch)
            max_trg_size = max(trg_size_batch)
            # max_src_size = self.params.MAX_LEN_S
            # max_trg_size = self.params.MAX_LEN_Q

            for i in range(self.start,self.end):
                index = self.indexs[i]
                s = self.src_input[index]
                q = self.trg_target[index]
                pad_s = [0]*(max_src_size-len(s))
                pad_q = [0]*(max_trg_size-len(q))
                src_input_batch.append(s+pad_s)
                trg_target_batch.append(q+pad_q)
                trg_input_batch.append([self.params.SOS_ID]+q[:-1]+pad_q)
            
            self.start += self.batch_size
            yield (len(src_input_batch),np.array(src_input_batch),np.array(src_size_batch),\
                np.array(trg_input_batch),np.array(trg_target_batch),np.array(trg_size_batch))
        