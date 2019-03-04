#coding:utf-8
from collections import Counter
from operator import itemgetter
from tqdm import tqdm
import json
import numpy as np
import os

src = "../data2.0/src-train.txt"#输入文件
tgt = "../data2.0/tgt-train.txt"#输入文件
para = "../data2.0/para-train.txt"#输入文件

src_vocab = "../data2.0/src_vocab_file"#输出文件
tgt_vocab = "../data2.0/tgt_vocab_file"#输出文件
para_vocab = "../data2.0/para_vocab_file"#输出文件

src_w2i = "../data2.0/src_w2i_file"#输出文件
tgt_w2i = "../data2.0/tgt_w2i_file"#输出文件
para_w2i = "../data2.0/para_w2i_file"#输出文件

src_i2w = "../data2.0/src_i2w_file"#输出文件
tgt_i2w = "../data2.0/tgt_i2w_file"#输出文件
para_i2w = "../data2.0/para_i2w_file"#输出文件

src_vocab_size = 45000
tgt_vocab_size = 28000
para_vocab_size = 45000

def generate_vocab(input_filename,vocab_size,output1,output2,output3):
	counter = Counter()
	with open(input_filename,'r') as f:
		all_sentences = f.readlines()
		print("creating vocab...")
		for sentence in tqdm(all_sentences):
			if sentence:
				word_list = sentence.strip().split()
				for word in word_list:
					counter[word] += 1

	sorted_counter  = sorted(counter.items(),key=itemgetter(1),reverse=True)
	sorted_words = ['<pad>','<unk>','<sos>','<eos>'] + [x[0] for x in sorted_counter]
	if len(sorted_words) > vocab_size:
		sorted_words = sorted_words[:vocab_size]


	with open(output1,'w') as f:
		for i in range(vocab_size):
			f.write(sorted_words[i]+"\n")
	print("vocab done...")

	with open(output2,'w') as f:
		d = dict()
		for i in range(vocab_size):
			d[sorted_words[i]] = i
		json.dump(d,f)
	print("w2i done...")

	with open(output3,'w') as f:
		d = dict()
		for i in range(vocab_size):
			d[i] = sorted_words[i]
		json.dump(d,f)
	print("i2w done...")

if __name__ == '__main__':
	generate_vocab(src,src_vocab_size,src_vocab,src_w2i,src_i2w)
	generate_vocab(tgt,tgt_vocab_size,tgt_vocab,tgt_w2i,tgt_i2w)
	generate_vocab(para,para_vocab_size,para_vocab,para_w2i,para_i2w)