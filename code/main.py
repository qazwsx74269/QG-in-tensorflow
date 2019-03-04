#coding:utf-8
import tensorflow as tf
import train_model
import eval_model
import eval_model_beamsearch
import fire
from tqdm import tqdm
from config import opt
from dataset_process import MakeSrcTrgDataset
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import os
import json


def eval(**kwargs):
	opt.parse(kwargs)
	if opt.GENERATE:
		tf.reset_default_graph()
		data_test = MakeSrcTrgDataset(opt.SRC_TEST_DATA,opt.TRG_TEST_DATA,opt.BATCH_SIZE,isTrain=False)
		iterator_test = data_test.make_initializable_iterator()
		(src2,src_size2),(trg_input2,trg_label2,trg_size2) = iterator_test.get_next()

		# nlp = StanfordCoreNLP("../../stanford-corenlp-full-2018-10-05",lang='en')
		
		if not opt.BEAM_SEARCH:
			with tf.variable_scope("QGModel",reuse=None):
				model = eval_model.QGModel(opt)
		else:
			with tf.variable_scope("QGModel",reuse=None):
				model = eval_model_beamsearch.QGModel(opt)
			
		output_op = model.forward(src2,src_size2)

		with open(opt.SRC_ID2WORD,'r') as f:
			i2w = json.load(f)
		with open(opt.TGT_ID2WORD,'r') as f:
			i2w_ = json.load(f)
		f1 = open(opt.TEST_SENTENCES,'w')
		f2 = open(opt.TEST_QUESTIONS,'w')
		f3 = open(opt.PREDICTED_QUESTIONS,'w')

		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7,allow_growth=True)
		session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		include = ['encoder/bidirectional_rnn/fw/multi_rnn_cell/cell_1/basic_lstm_cell/bias', 
		'encoder/bidirectional_rnn/bw/multi_rnn_cell/cell_1/basic_lstm_cell/kernel',
		'encoder/bidirectional_rnn/fw/multi_rnn_cell/cell_1/basic_lstm_cell/kernel',
		'encoder/bidirectional_rnn/bw/multi_rnn_cell/cell_1/basic_lstm_cell/bias',
		'encoder/bidirectional_rnn/bw/multi_rnn_cell/cell_0/basic_lstm_cell/bias',
		'QGModel/softmax_weight',
		'decoder/rnn/attention_wrapper/bahdanau_attention/query_layer/kernel',
		'decoder/rnn/attention_wrapper/multi_rnn_cell/cell_0/basic_lstm_cell/kernel',
		'decoder/memory_layer/kernel',
		'encoder/bidirectional_rnn/fw/multi_rnn_cell/cell_0/basic_lstm_cell/kernel',
		'encoder/bidirectional_rnn/fw/multi_rnn_cell/cell_0/basic_lstm_cell/bias',
		'decoder/rnn/attention_wrapper/multi_rnn_cell/cell_1/basic_lstm_cell/bias',
		'encoder/bidirectional_rnn/bw/multi_rnn_cell/cell_0/basic_lstm_cell/kernel',
		'QGModel/softmax_bias',
		'decoder/rnn/attention_wrapper/attention_layer/kernel',
		'decoder/rnn/attention_wrapper/multi_rnn_cell/cell_0/basic_lstm_cell/bias',
		'decoder/rnn/attention_wrapper/multi_rnn_cell/cell_1/basic_lstm_cell/kernel',
		'decoder/rnn/attention_wrapper/bahdanau_attention/attention_v',
		]
		variables_to_restore = tf.contrib.slim.get_variables_to_restore(include=include)
		saver = tf.train.Saver(variables_to_restore)
		saver.restore(session,opt.CHECKPOINT_PATH)
		session.run(iterator_test.initializer)
		count = 0
		while True:
			try:
				src,tgt,output = session.run([src2,trg_input2,output_op])   
				src_sentence = " ".join([i2w[str(i)] for i in np.squeeze(src)][:-1])
				tgt_question = " ".join([i2w_[str(i)] for i in np.squeeze(tgt)][1:])
				predicted_question = " ".join([i2w_[str(i)] for i in np.squeeze(output)][1:-1])
				# print("predicted question:",predicted_question)
				f1.write(src_sentence+"\n")
				f2.write(tgt_question+"\n")
				f3.write(predicted_question+"\n")
				count = count + 1
				print(count)
			except tf.errors.OutOfRangeError:
				break
				
		session.close()
		# nlp.close()
		f1.close()
		f2.close()
		f3.close()

	os.system("python eval.py"+" -out "+opt.PREDICTED_QUESTIONS+\
			" -src "+opt.TEST_SENTENCES+\
			" -tgt "+opt.TEST_QUESTIONS)

def main(**kwargs):
	opt.parse(kwargs)
	initializer = tf.random_uniform_initializer(-0.05,0.05)
	with tf.variable_scope("QGModel",reuse=tf.AUTO_REUSE,initializer=initializer):
		model = train_model.QGModel(opt)
	data = MakeSrcTrgDataset(opt.SRC_TRAIN_DATA,opt.TRG_TRAIN_DATA,opt.BATCH_SIZE)
	data_eval = MakeSrcTrgDataset(opt.SRC_DEV_DATA,opt.TRG_DEV_DATA,opt.BATCH_SIZE,isTrain=False)
	iterator = data.make_initializable_iterator()
	iterator_eval = data_eval.make_initializable_iterator()
	(src,src_size),(trg_input,trg_label,trg_size) = iterator.get_next()
	(src1,src_size1),(trg_input1,trg_label1,trg_size1) = iterator_eval.get_next()
	cost_op,train_op = model.forward(src,src_size,trg_input,trg_label,trg_size)#inference + backpropagation
	cost1 = tf.summary.scalar("training stage per token cost",cost_op)  
	cost_op_eval,_= model.forward(src1,src_size1,trg_input1,trg_label1,trg_size1)#inference
	cost2 = tf.summary.scalar("evaluation stage per token cost",cost_op_eval)   
	saver = tf.train.Saver()
	step = 0
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7,allow_growth=True)
	session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	# session_eval = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	writer = tf.summary.FileWriter(opt.SUMMARY_DIR,session.graph)
	eval_writer = tf.summary.FileWriter(opt.SUMMARY_EVAL_DIR,session.graph)
	# merged = tf.summary.merge_all()
	session.run(tf.global_variables_initializer())

	for i in range(opt.EPOCHS):
		print("In iteration: %d"%(i+1))
		session.run(iterator.initializer)
		if i == 7:
			model.params.LEARNING_RATE /= 2.0 
		# step = run_epoch(cost1,cost2,writer,eval_writer,cost_op_eval,session,cost_op,train_op,saver,step)
		while True:#training stage
			try:
				writer.add_summary(session.run(cost1),step)
				cost,_ = session.run([cost_op,train_op])
				if step%10 == 0:
					print("steps %d, per token cost is %.3f"%(step,cost))

				if step%200 == 0:
					session.run(iterator_eval.initializer)
					# saver.restore(session,tf.train.latest_checkpoint(opt.CHECKPOINT_DIR))
					total_cost = []
					while True:#validation stage
						try:
							eval_writer.add_summary(session.run(cost2),step)    
							cost_eval = session.run(cost_op_eval)
							total_cost.append(cost_eval)
						except tf.errors.OutOfRangeError:
							# print(total_cost)
							print("steps: %d eval_average_loss: %.4f"%(step,np.mean(total_cost)))
							break
					saver.save(session,opt.CHECKPOINT_PATH,global_step=step)
				step += 1
			except tf.errors.OutOfRangeError:
				break
	session.close()
	writer.close()

if __name__ == '__main__':
	fire.Fire()