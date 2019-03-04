import pickle
import json
import numpy as np
from tqdm import tqdm
import unicodedata
import string
# from nltk import word_tokenize
# from nltk.tokenize.moses import MosesDetokenizer
import argparse
import random
# from gensim.scripts.glove2word2vec import glove2word2vec
# from gensim.models import KeyedVectors
# import torch
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP("../../stanford-corenlp-full-2018-10-05",lang='en')#结尾一定要加nlp.close()
props={'annotators': 'ssplit','pipelineLanguage':'en','outputFormat':'json'}
# detokenizer = MosesDetokenizer()
all_letters = string.printable
# max_length_of_passage = 400

data_root = '../data2.0/'

def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		and c in all_letters)

def sentence_split_and_tokenization(text):
	json_str = nlp.annotate(text, properties=props)
	d = json.loads(json_str)
	sentences_count = len(d["sentences"])
	sentences = []
	for i in range(sentences_count):
		sentence_d = d["sentences"][i]
		tokens = sentence_d['tokens']
		words_count = len(tokens)
		words = []      
		for j in range(words_count):
			word_d = tokens[j]
			words.append(word_d["word"])
		sentences.append(words)
	return sentences

def preprocess_json_to_get_data(filepath):

	with open(filepath, encoding='utf-8') as q:
		data = json.load(q)

	paragraphs = []
	sentences = []
	questions = []

	for i in tqdm(range(len(data['data']))):
		for j in range(len(data['data'][i]['paragraphs'])):

			passage = unicodeToAscii(data['data'][i]['paragraphs'][j]['context'])
			while '\n' in passage:
				index = passage.index('\n')
				passage = passage[0:index] + passage[index + 1:]

			sentences_list = sentence_split_and_tokenization(passage)
			paragraph_list = []
			for sentence_list in sentences_list:
				paragraph_list += sentence_list

			for k in range(len(data['data'][i]['paragraphs'][j]['qas'])):

				question = unicodeToAscii(data['data'][i]['paragraphs'][j]['qas'][k]['question'])
				# id1 = data['data'][i]['paragraphs'][j]['qas'][k]['id']
				# all_ids_in_data.append(id1)

				if data['data'][i]['paragraphs'][j]['qas'][k]['is_impossible'] == True:
					answer_key = 'plausible_answers'
				else:
					answer_key = 'answers'

				# all_starts, all_ends, all_answers = [], [], []
				for l in range(len(data['data'][i]['paragraphs'][j]['qas'][k][answer_key])):
					answer = unicodeToAscii(data['data'][i]['paragraphs'][j]['qas'][k][answer_key][l]['text'])
					# start = data['data'][i]['paragraphs'][j]['qas'][k][answer_key][l]['answer_start']
					# end = data['data'][i]['paragraphs'][j]['qas'][k][answer_key][l]['answer_start'] + len(answer)

					while '\n' in answer:
						index = answer.index('\n')
						answer = answer[0:index] + answer[index + 1:]

					temp_words_list = []
					concat_sentences_num = 0
					for words_list in sentences_list:
						sentence_str = ' '.join(words_list)
						if  answer in sentence_str:
							temp_words_list += words_list
							concat_sentences_num += 1
					if concat_sentences_num>0:
						paragraphs.append(paragraph_list)
						sentences.append(temp_words_list)
						questions.append(nlp.word_tokenize(question))
						# print("csn: ",concat_sentences_num)

	return paragraphs,sentences,questions

def write_to_file(data,filename):
	with open(filename,'w') as f:
		for line in data:
			data_str = ' '.join(line)
			f.write(data_str.lower()+"\n")

def generate_train_dev_test(json_train_file,json_dev_file):
	p1,s1,q1 = preprocess_json_to_get_data(json_train_file)
	p2,s2,q2 = preprocess_json_to_get_data(json_dev_file)
	p = p1 + p2
	s = s1 + s2
	q = q1 + q2
	nums = len(p)
	assert nums == len(s)
	assert nums == len(q)
	print(nums)
	indexes = list(range(nums))
	random.shuffle(indexes)
	train_nums = int(nums*0.8)
	dev_nums = int(nums*0.1)
	test_nums = nums - dev_nums - train_nums 
	for data,data_type in zip([p,s,q],['para','src','tgt']):
		write_to_file(list(np.take(data,indexes[:train_nums])),data_root+data_type+'-train.txt')
		write_to_file(list(np.take(data,indexes[train_nums:train_nums+dev_nums])),data_root+data_type+'-dev.txt')
		write_to_file(list(np.take(data,indexes[train_nums+dev_nums:])),data_root+data_type+'-test.txt')



if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Question Answering Data Preprocess')
	parser.add_argument('--train_json', type=str, required = True, help='Training JSON File for SQuAD 2.0')
	parser.add_argument('--dev_json', type=str, required = True, help='Development JSON File for SQuAD 2.0 ')
	# parser.add_argument('--glove_file', type=str, required = True, help='Glove Embedding File')
	args = parser.parse_args()

	json_dev_file = args.dev_json       ##  './json_data_files/dev-v1.1.json'
	json_train_file = args.train_json   ##  './json_data_files/train-v1.1.json'
	
	generate_train_dev_test(json_train_file,json_dev_file)
	nlp.close()