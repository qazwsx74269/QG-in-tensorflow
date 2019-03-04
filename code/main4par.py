#coding:utf-8
import tensorflow as tf
import model5par
import fire
from tqdm import tqdm
from config import opt
from dataset2 import Batcher
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import os
import json
import eval

def replace_unk(alignment_history,output,src_input,src_size,beam_size):
	shapes = alignment_history.shape
	decode_size = shapes[0]
	assert beam_size == shapes[1]
	assert src_size == shapes[2]
	assert output.shape[0] == shapes[0]#decode_size
	replaced_src_positions = np.argmax(alignment_history,axis=2)
	replaced_src_ids = np.take(src_input,replaced_src_positions)

	replaced_output = np.zeros_like(output)
	is_replace = np.zeros_like(output)
	for i in range(decode_size):
		for j in range(beam_size):
			if output[i,j] != opt.UNK_ID:
				replaced_output[i,j] = output[i,j]
			else:
				# print("replace_unk")
				is_replace[i,j] = 1
				replaced_output[i,j] = replaced_src_ids[i,j]
	return replaced_output,is_replace


def generation(**kwargs):
	opt.parse(kwargs)
	# tf.reset_default_graph()
	# nlp = StanfordCoreNLP("../../stanford-corenlp-full-2018-10-05",lang='en')
	with open(opt.SRC_ID2WORD,'r') as f:
			i2w = json.load(f)
	with open(opt.TGT_ID2WORD,'r') as f:
			i2w_ = json.load(f)
	f1 = open(opt.TEST_SENTENCES,'w')
	f2 = open(opt.TEST_QUESTIONS,'w')
	

	with tf.variable_scope("QGModel",reuse=None):
		# opt.BATCH_SIZE = 1
		batcher_test =  Batcher(opt)
		model = model5par.QGModel(opt)	

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7,allow_growth=True)
	session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	saver = tf.train.Saver()
	saver.restore(session,tf.train.latest_checkpoint(opt.PARA_CHECKPOINT_DIR))
	count = 0
	outputs = []
	is_replaces = []
	for (batch_size,src_input,src_size,para_input,para_size,trg_input,trg_label,trg_size) in tqdm(batcher_test.get_batch()):
		alignment_history,output = session.run([model.alignment_history,model.decoder_predict_decode],
			feed_dict={model.batch_size:batch_size,
				model.src_input:src_input,
				model.src_size:src_size,
				model.para_input:para_input,
				model.para_size:para_size,
				model.trg_input:trg_input,
				model.trg_label:trg_label,
				model.trg_size:trg_size,
				model.keep_prob_placeholder: 1.0}) 
		beam_size =  model.params.BEAM_SIZE
		src_sentence = " ".join([i2w[str(i)] for i in np.squeeze(src_input)[:-1]])
		tgt_question = " ".join([i2w_[str(i)] for i in np.squeeze(trg_input)[1:]])
		f1.write(src_sentence+"\n")
		f2.write(tgt_question+"\n")
		count = count + 1
		# print(count)
		output = np.squeeze(output,axis=0)#remove batch-size
		if model.params.REPLACE_UNK:
			output,is_replace = replace_unk(alignment_history,output,src_input,src_size,beam_size)
			outputs.append(output)
			is_replaces.append(is_replace)
		else:
			outputs.append(output)

	f1.close()
	f2.close()
	np.save("outputs.npy",np.array(outputs))
	np.save("is_replaces.npy",is_replaces)
	session.close()

def evaluation(**kwargs):
	opt.parse(kwargs)
	with open(opt.SRC_ID2WORD,'r') as f:
			i2w = json.load(f)
	with open(opt.TGT_ID2WORD,'r') as f:
			i2w_ = json.load(f)
	beam_size = opt.BEAM_SIZE
	outputs = np.load("outputs.npy")
	is_replaces = np.load("is_replaces.npy")
	count = outputs.shape[0]
	for j in range(beam_size):
		f3 = open(opt.PREDICTED_QUESTIONS+str(j)+".txt",'w')
		for k in range(count):
			temp = []
			for index,i in enumerate(outputs[k][:-1,j]):
				if i!=opt.EOS_ID:
					if len(is_replaces)>0 and is_replaces[k][index,j]==1 and opt.REPLACE_UNK:
						if i2w[str(i)] == "<unk>":
							print("unk")
						temp.append(i2w[str(i)])
					else:
						temp.append(i2w_[str(i)])
				predicted_question = " ".join(temp)
			f3.write(predicted_question+"\n")
		
		f3.close()
		
	for j in range(beam_size):
		print("\nbeam %d's result"%(j))
		eval.eval(opt.PREDICTED_QUESTIONS+str(j)+".txt",opt.TEST_SENTENCES,opt.TEST_QUESTIONS)

def train_and_validation(**kwargs):
	opt.parse(kwargs)
	initializer = tf.random_uniform_initializer(-0.05,0.05)
	with tf.variable_scope("QGModel",initializer=initializer):
		opt.MODE = 'train'
		batcher = Batcher(opt)
		train_model = model5par.QGModel(opt)
		opt.MODE = 'val'
		batcher_val = Batcher(opt)

	saver = tf.train.Saver()
	step = 0
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7,allow_growth=True)
	session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	writer = tf.summary.FileWriter(opt.SUMMARY_DIR,session.graph)
	eval_writer = tf.summary.FileWriter(opt.SUMMARY_EVAL_DIR,session.graph)
	session.run(tf.global_variables_initializer())
	step = 0
	minloss = 999999.0
	for i in range(opt.EPOCHS):
		if i==7:
			train_model.params.LEARNING_RATE /= 2.0
		for (batch_size,src_input,src_size,para_input,para_size,trg_input,trg_label,trg_size)  in batcher.get_batch():
			summary,cost,_ = session.run([train_model.summary,train_model.loss,train_model.train_op],
				feed_dict={train_model.batch_size:batch_size,
					train_model.src_input:src_input,
					train_model.src_size:src_size,
					train_model.para_input:para_input,
					train_model.para_size:para_size,
					train_model.trg_input:trg_input,
					train_model.trg_label:trg_label,
					train_model.trg_size:trg_size,
					train_model.keep_prob_placeholder: opt.KEEP_PROB})
			writer.add_summary(summary,step)
		
			if step%10 == 0:
				print("epochs %d, steps %d, per token cost is %.3f"%(i,step,cost))
			if step%200 == 0:
				total_steps = 0
				total_loss = 0
				for (batch_size,src_input,src_size,para_input,para_size,trg_input,trg_label,trg_size) in batcher_val.get_batch():
						summary,cost = session.run([train_model.summary,train_model.loss],
							feed_dict={train_model.batch_size:batch_size,
								train_model.src_input:src_input,
								train_model.src_size:src_size,
								train_model.para_input:para_input,
								train_model.para_size:para_size,
								train_model.trg_input:trg_input,
								train_model.trg_label:trg_label,
								train_model.trg_size:trg_size,
								train_model.keep_prob_placeholder: 1.0})
						total_steps += 1
						total_loss += cost

				test_summary = tf.Summary()
				lossval = test_summary.value.add()
				lossval.tag = 'val loss'
				lossval.simple_value = total_loss*1.0/ total_steps
				print("steps: %d eval_per token loss: %.4f"%(step,lossval.simple_value))
				eval_writer.add_summary(test_summary,step)
				if lossval.simple_value < minloss:
					minloss = lossval.simple_value
					saver.save(session,opt.PARA_CHECKPOINT_PATH,global_step=step)
			step += 1

	session.close()
	writer.close()
	eval_writer.close()

if __name__ == '__main__':
	fire.Fire()
