#coding:utf-8
from collections import Counter
from operator import itemgetter
from tqdm import tqdm
import json
import numpy as np
import os

para = "../data2.0/para-train.txt"#输入文件
src = "../data2.0/src-train.txt"#输入文件
tgt = "../data2.0/tgt-train.txt"#输入文件
para_word2id = "../data2.0/para_w2i_file"#输入文件
src_word2id = "../data2.0/src_w2i_file"#输入文件
tgt_word2id = "../data2.0/tgt_w2i_file"#输入文件
para_id = para + ".id"#输出文件
src_id = src + ".id"#输出文件
tgt_id = tgt + ".id"#输出文件

def generate_id(src,src_word2id,tgt,tgt_word2id,src_id,tgt_id):
	f1 = open(src,'r')
	f2 = open(tgt,'r')
	print("converting src...")

	with open(src_id,'w') as fs:
		lines = f1.readlines()
		with open(src_word2id,'r') as f:
			w2i = json.load(f)
		for line in tqdm(lines):
			if line:
				splits = line.strip().split() + ["<eos>"]
				ids = [str(w2i[w]) if w in w2i else str(w2i["<unk>"]) for w in splits]
				id_str = ' '.join(ids)
				fs.write(id_str+"\n")

	print("converting tgt...")
	with open(tgt_id,'w') as ft:
		lines = f2.readlines()
		with open(tgt_word2id,'r') as f:
			w2i = json.load(f)
		for line in tqdm(lines):
			if line:
				splits = line.strip().split() + ["<eos>"]
				ids = [str(w2i[w]) if w in w2i else str(w2i["<unk>"]) for w in splits]
				id_str = ' '.join(ids)
				ft.write(id_str+"\n")

	f1.close()
	f2.close()

def generate_para_id(para,para_word2id,para_id):
	f1 = open(para,'r')
	print("converting para...")

	with open(para_id,'w') as fs:
		lines = f1.readlines()
		with open(para_word2id,'r') as f:
			w2i = json.load(f)
		for line in tqdm(lines):
			if line:
				splits = line.strip().split() + ["<eos>"]
				ids = [str(w2i[w]) if w in w2i else str(w2i["<unk>"]) for w in splits]
				id_str = ' '.join(ids)
				fs.write(id_str+"\n")
	f1.close()

if __name__ == '__main__':
	src = "../data2.0/src-train.txt"#输入文件
	tgt = "../data2.0/tgt-train.txt"#输入文件
	src_id = src + ".id"#输出文件
	tgt_id = tgt + ".id"#输出文件
	generate_id(src,src_word2id,tgt,tgt_word2id,src_id,tgt_id)
	src = "../data2.0/src-dev.txt"#输入文件
	tgt = "../data2.0/tgt-dev.txt"#输入文件
	src_id = src + ".id"#输出文件
	tgt_id = tgt + ".id"#输出文件
	generate_id(src,src_word2id,tgt,tgt_word2id,src_id,tgt_id)
	src = "../data2.0/src-test.txt"#输入文件
	tgt = "../data2.0/tgt-test.txt"#输入文件
	src_id = src + ".id"#输出文件
	tgt_id = tgt + ".id"#输出文件
	generate_id(src,src_word2id,tgt,tgt_word2id,src_id,tgt_id)

	generate_para_id("../data2.0/para-train.txt",para_word2id,"../data2.0/para-train.txt.id")
	generate_para_id("../data2.0/para-dev.txt",para_word2id,"../data2.0/para-dev.txt.id")
	generate_para_id("../data2.0/para-test.txt",para_word2id,"../data2.0/para-test.txt.id")