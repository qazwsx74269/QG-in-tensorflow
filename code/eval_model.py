#coding:utf-8
import tensorflow as tf
import numpy as np


class QGModel(object):
	def __init__(self,params):
		self.params = params
		self.enc_cell_fw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(params.HIDDEN_SIZE)\
			for _ in range(params.NUM_LAYERS)])
		self.enc_cell_bw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(params.HIDDEN_SIZE)\
			for _ in range(params.NUM_LAYERS)])
		self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(params.HIDDEN_SIZE*2)\
			for _ in range(params.NUM_LAYERS)])
		self.softmax_weight = tf.get_variable("softmax_weight",[params.HIDDEN_SIZE*2,params.TGT_VOCAB_SIZE])
		self.softmax_bias = tf.get_variable("softmax_bias",[params.TGT_VOCAB_SIZE])
		self.src_pretrained_embedding = tf.convert_to_tensor(np.load(self.params.SRC_ID2EMBEDDING),dtype=tf.float32)
		self.tgt_pretrained_embedding = tf.convert_to_tensor(np.load(self.params.TGT_ID2EMBEDDING),dtype=tf.float32)

	def forward(self,src_input,src_size):#greedy search
		batch_size = tf.shape(src_input)[0]
		src_emb = tf.nn.embedding_lookup(self.src_pretrained_embedding,src_input)
		print("src_emb:",src_emb.get_shape().as_list())
		with tf.variable_scope("encoder"):
			enc_outputs,enc_state = tf.nn.bidirectional_dynamic_rnn(
				self.enc_cell_fw,self.enc_cell_bw,src_emb,src_size,dtype=tf.float32)
			enc_outputs = tf.concat([enc_outputs[0],enc_outputs[1]],-1)
		# print(enc_outputs.get_shape().as_list())

		# lstm_state_as_tensor_shape = [self.params.NUM_LAYERS, 2, batch_size, self.params.HIDDEN_SIZE]
		# initial_state = tf.zeros(lstm_state_as_tensor_shape)
		# unstack_state = tf.unstack(enc_state, axis=0)
		# tuple_state = tuple([([enc_state[0][idx][0], enc_state[1][idx][1]],-1) for idx in range(self.params.NUM_LAYERS)])
			tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(tf.concat([enc_state[0][idx].c,enc_state[1][idx].c],-1),\
				tf.concat([enc_state[0][idx].h,enc_state[1][idx].h],-1)) for idx in range(self.params.NUM_LAYERS)])
		# print(enc_state[0][0].get_shape().as_list())

		trg_ids = tf.TensorArray(dtype=tf.int32,size=0,dynamic_size=True,clear_after_read=False)
		trg_ids = trg_ids.write(0,tf.fill([batch_size],self.params.SOS_ID))
		# output_array = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True,clear_after_read=False)
		with tf.variable_scope("decoder"):
			self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.params.HIDDEN_SIZE*2,\
				enc_outputs,memory_sequence_length=src_size)
		with tf.variable_scope("decoder/rnn/attention_wrapper"):
			self.attention_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_cell,self.attention_mechanism,\
				attention_layer_size=self.params.HIDDEN_SIZE*2)
			state = self.attention_cell.zero_state(batch_size=batch_size,dtype=tf.float32).clone(cell_state=tuple_state)
			##NOTE Please see the initializer documentation for details of how to call zero_state if using an AttentionWrapper with a BeamSearchDecoder.
			init_loop_var = (state,trg_ids,0)

			def continue_loop_condition(state,trg_ids,step):
				return tf.reduce_all(tf.logical_and(tf.not_equal(trg_ids.read(step),self.params.EOS_ID),\
					tf.less(step,self.params.MAX_DEC_LEN-1)))

			def loop_body(state,trg_ids,step):
				trg_input = trg_ids.read(step)
				trg_emb = tf.nn.embedding_lookup(self.tgt_pretrained_embedding,trg_input)
				print("trg_emb:",trg_emb.get_shape().as_list())
				dec_outputs,next_state = self.attention_cell.call(state=state,inputs=trg_emb)#!!!!!!!!!!!
				output = tf.reshape(dec_outputs,[-1,self.params.HIDDEN_SIZE*2])
				logits = tf.matmul(output,self.softmax_weight) + self.softmax_bias
				next_id = tf.argmax(logits,axis=1,output_type=tf.int32)
				trg_ids = trg_ids.write(step+1,next_id)
				return next_state,trg_ids,step+1

			state,trg_ids,step = tf.while_loop(continue_loop_condition,loop_body,init_loop_var)
			return trg_ids.stack()
			
	# def forward_(self,src_input,src_size):#beam search
	# 	batch_size = tf.shape(src_input)[0]
	# 	src_emb = tf.nn.embedding_lookup(self.src_pretrained_embedding,src_input)
	# 	print("src_emb:",src_emb.get_shape().as_list())
	# 	with tf.variable_scope("encoder"):
	# 		enc_outputs,enc_state = tf.nn.bidirectional_dynamic_rnn(
	# 			self.enc_cell_fw,self.enc_cell_bw,src_emb,src_size,dtype=tf.float32)
	# 		enc_outputs = tf.concat([enc_outputs[0],enc_outputs[1]],-1)
	# 	tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(tf.concat([enc_state[0][idx].c,enc_state[1][idx].c],-1),\
	# 		tf.concat([enc_state[0][idx].h,enc_state[1][idx].h],-1)) for idx in range(self.params.NUM_LAYERS)])

	# 	encoder_outputs = tf.contrib.seq2seq.tile_batch(enc_outputs,multiplier=self.params.BEAM_SIZE)
	# 	encoder_state = tf.contrib.framework.nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.params.BEAM_SIZE), tuple_state)
	# 	encoder_inputs_length = tf.contrib.seq2seq.tile_batch(src_size, multiplier=self.params.BEAM_SIZE)
	# 	with tf.variable_scope("decoder"):
	# 		self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.params.HIDDEN_SIZE*2,\
	# 			encoder_outputs,memory_sequence_length=encoder_inputs_length)
	# 	with tf.variable_scope("decoder/rnn/attention_wrapper"):
	# 		self.attention_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_cell,self.attention_mechanism,\
	# 			attention_layer_size=self.params.HIDDEN_SIZE*2)
	# 		state = self.attention_cell.zero_state(batch_size=batch_size*self.params.BEAM_SIZE,dtype=tf.float32).clone(cell_state=encoder_state)
	# 	print("use beam search decoding...")
	# 	start_tokens = tf.ones([batch_size, ], tf.int32) * self.params.SOS_ID
	# 	end_token = self.params.EOS_ID
	# 	inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=self.attention_cell, embedding=self.tgt_pretrained_embedding,
	# 																		 start_tokens=start_tokens, end_token=end_token,
	# 																		 initial_state=state,
	# 																		 beam_width=self.params.BEAM_SIZE,
	# 																		 output_layer=None)
	# 	decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
	# 															maximum_iterations=self.params.MAX_DEC_LEN-1)
	# 	# 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，
	# 	# 对于不使用beam_search的时候，它里面包含两项(rnn_outputs, sample_id)
	# 	# rnn_output: [batch_size, decoder_targets_length, vocab_size]
	# 	# sample_id: [batch_size, decoder_targets_length], tf.int32

	# 	# 对于使用beam_search的时候，它里面包含两项(predicted_ids, beam_search_decoder_output)
	# 	# predicted_ids: [batch_size, decoder_targets_length, beam_size],保存输出结果
	# 	# beam_search_decoder_output: BeamSearchDecoderOutput instance namedtuple(scores, predicted_ids, parent_ids)
	# 	# 所以对应只需要返回predicted_ids或者sample_id即可翻译成最终的结果
	# 	dec_outputs = decoder_outputs.beam_search_decoder_output.scores
	# 	output = tf.reshape(dec_outputs,[-1,self.params.HIDDEN_SIZE*2])
	# 	logits = tf.matmul(output,self.softmax_weight) + self.softmax_bias

	# 	ids = tf.argmax(tf.reshape(logits,[-1,self.params.TGT_VOCAB_SIZE]),axis=-1,output_type=tf.int32)
	# 	ids = tf.reshape(ids,[-1,self.params.BEAM_SIZE])
	# 	return ids
