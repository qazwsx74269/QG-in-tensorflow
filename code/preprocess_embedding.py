from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
from six.moves import zip
from six.moves import cPickle

import numpy as np
from tqdm import tqdm

TRANSLATE = {
    "-lsb-" : "[",
    "-rsb-" : "]",
    "-lrb-" : "(",
    "-rrb-" : ")",
    "-lcb-" : "{",
    "-rcb-" : "}",
    "-LSB-" : "[",
    "-RSB-" : "]",
    "-LRB-" : "(",
    "-RRB-" : ")",
    "-LCB-" : "{",
    "-RCB-" : "}",
}

def parse_args(description="I am copying"):
	import argparse
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument("--embedding",type=str,default="../glove.840B.300d.txt",required=True)
	parser.add_argument("--dict",type=str,default="../data2.0/para_vocab_file",required=True)
	parser.add_argument("--output",type=str,default="../data2.0/para_embedding_file",required=True)
	parser.add_argument("--seed",type=int,default=19941109)
	args = parser.parse_args()
	np.random.seed(args.seed)
	return args

def main():
	args = parse_args()
	word2embedding = {}
	dimension = 300
	with open(args.embedding,"r") as input_file:
		print("processing glove...")
		for line in tqdm(input_file):
			line = line.strip().split()
			try:
				l = len(line)
				if l>301:
					print(' '.join(line[0:(l-300)]))
					word2embedding[' '.join(line[0:(l-300)])] = np.array(list(map(float,line[(l-300):])))#phrase比如contact name@domain.com
				else:
					word2embedding[line[0]] = np.array(list(map(float,line[1:])))#word
			except:
				print(line[1:])
				break
		
	with open(args.dict,"r") as input_file:
		words = [line.strip().split()[0] for line in input_file]

	embedding = np.random.uniform(low=-1.0/3,high=1.0/3,size=(len(words),dimension))
	embedding = np.asarray(embedding,dtype=np.float32)
	unknown_count = 0
	print("processing vocab embedding...")
	embedding[0] = 0.0 
	embedding[1] = word2embedding['UNKNOWN']
	embedding[2] = word2embedding['<s>']
	embedding[3] = word2embedding['EOS']
	special_list = ['<pad>','<unk>','<sos>','<eos>']
	for i,word in tqdm(enumerate(words)):
		if word in TRANSLATE:
			word = TRANSLATE[word]
		done = False
		for w in (word,word.upper(),word.lower()):
			if w in word2embedding and w not in special_list:
				embedding[i] = word2embedding[w]
				done = True
				break
		if word in special_list:
			done = True
		if not done:
			print("Unknown word: %s"%(word,))
			unknown_count += 1
	np.save(args.output,embedding)
	print("Total unknown: %d"%(unknown_count,))

if __name__ == '__main__':
	main()