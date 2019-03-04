#coding:utf-8
from collections import Counter
from operator import itemgetter
from tqdm import tqdm
import json
import numpy as np
import os

src = "../data/src-train.txt"#输入文件
tgt = "../data/tgt-train.txt"#输入文件
pretrained_embedding_data = "../glove.840B.300d.txt"#输入文件

vocab = "../vocab_file"#输出文件
word2id = "../word2id"#输出文件
id2word = "../id2word"#输出文件
src_id = src + ".id"#输出文件
tgt_id = tgt + ".id"#输出文件
glove_id2embedding = "../glove_id2embedding"#输出文件
id2embedding = "../id2embedding"#输出文件

vocab_size = 45000
#第一步建库,建立单词到id的映射，id到单词的映射********************************************************************
if not os.path.exists(vocab):
	counter = Counter()
	f1 = open(src,'r')
	f2 = open(tgt,'r')
	all_sentences = f1.readlines()+f2.readlines()
	print("creating vocab...")
	for sentence in tqdm(all_sentences):
		if sentence:
			word_list = sentence.strip().split()
			for word in word_list:
				counter[word] += 1

	f1.close()
	f2.close()

	sorted_counter  = sorted(counter.items(),key=itemgetter(1),reverse=True)
	sorted_words = ['<pad>','<unk>','<sos>','<eos>'] + [x[0] for x in sorted_counter]
	if len(sorted_words) > vocab_size:
		sorted_words = sorted_words[:vocab_size]


	f3 = open(vocab,'w')
	fwi = open(word2id,'w')
	fiw = open(id2word,'w')
	w2i = dict()
	i2w = dict()
	for i,word in enumerate(sorted_words):
		w2i[word] = i
		i2w[i] = word
		f3.write(word+"\n")
	json.dump(w2i,fwi)
	json.dump(i2w,fiw)

	fwi.close()
	fiw.close()
	f3.close()
else:
	print("loading vocab...")
	with open(word2id,'r') as f:
		w2i = json.load(f)
	with open(id2word,'r') as f:
		i2w = json.load(f)


#第二步将文本转成编号
if not os.path.exists(src_id):
	f4 = open(src,'r')
	f5 = open(tgt,'r')
	print("converting src...")
	with open(src_id,'w') as fs:
		lines = f4.readlines()
		for line in tqdm(lines):
			if line:
				splits = line.strip().split() + ["<eos>"]
				ids = [str(w2i[w]) if w in w2i else str(w2i["<unk>"]) for w in splits]
				id_str = ' '.join(ids)
				fs.write(id_str+"\n")

	print("converting tgt...")
	with open(tgt_id,'w') as ft:
		lines = f5.readlines()
		for line in tqdm(lines):
			if line:
				splits = line.strip().split() + ["<eos>"]
				ids = [str(w2i[w]) if w in w2i else str(w2i["<unk>"]) for w in splits]
				id_str = ' '.join(ids)
				ft.write(id_str+"\n")

	f4.close()
	f5.close()
else:
	print("it already exists!")
#第三步建立id到pretrained_embedding的映射********************************************************************

if not os.path.exists(id2embedding+".npy"):
	f6 = open(pretrained_embedding_data)
	lines = f6.readlines()
	embedding_dict = dict()

	print("glove word2embedding...")
	for line in tqdm(lines):
		splits = line.strip().split()
		# print(len(splits))
		try:
			l = len(splits)
			if l>301:
				print(' '.join(splits[0:(l-300)]))
				embedding_dict[' '.join(splits[0:(l-300)])] = np.array(list(map(float,splits[(l-300):])))#phrase比如contact name@domain.com
			else:
				embedding_dict[splits[0]] = np.array(list(map(float,splits[1:])))#word
		except:
			print(splits[1:])
			break
		# print(embedding_dict[splits[0]])
	id2embedding_dict = np.tile(embedding_dict['UNKNOWN'],[len(w2i),1])
	unk_num = 0

	print("id2embedding...")
	id2embedding_dict[w2i['<unk>']] = embedding_dict['UNKNOWN']
	id2embedding_dict[w2i['<sos>']] = embedding_dict['<s>']
	id2embedding_dict[w2i['<eos>']] = embedding_dict['EOS']
	for word in w2i.keys():
		if word in embedding_dict.keys():
			id2embedding_dict[w2i[word]] = embedding_dict[word]
		else:
			unk_num += 1

	print("UNKNOWN words count = %d" % unk_num)
	f6.close()
	# print("saving glove_id2embedding...")
	# np.save(glove_id2embedding,embedding_dict)
	print("saving id2embedding...")
	np.save(id2embedding,id2embedding_dict)