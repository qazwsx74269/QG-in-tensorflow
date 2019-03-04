#coding:utf-8
import tensorflow as tf
import numpy as np

def get_shape(tensor):
  """Returns static shape if available and dynamic shape otherwise."""
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
		  for s in zip(static_shape, dynamic_shape)]
  return dims

def batch_gather(tensor, indices):
  """Gather in batch from a tensor of arbitrary size.

  In pseudocode this module will produce the following:
  output[i] = tf.gather(tensor[i], indices[i])

  Args:
	tensor: Tensor of arbitrary size.
	indices: Vector of indices.
  Returns:
	output: A tensor of gathered values.
  """
  shape = get_shape(tensor)
  flat_first = tf.reshape(tensor, [shape[0] * shape[1]] + shape[2:])
  indices = tf.convert_to_tensor(indices)
  offset_shape = [shape[0]] + [1] * (indices.shape.ndims - 1)
  offset = tf.reshape(tf.range(shape[0]) * shape[1], offset_shape)
  output = tf.gather(flat_first, indices + offset)
  return output

import tensorflow as tf

def rnn_beam_search(batch_size,embedding,weight,bias,update_fn, state, sequence_length, beam_width,
					begin_token_id, end_token_id, name="rnn"):
	"""Beam-search decoder for recurrent models.

	Args:
	update_fn: Function to compute the next state and logits given the current
			   state and ids.
	initial_state: Recurrent model states.
	sequence_length: Length of the generated sequence.
	beam_width: Beam width.
	begin_token_id: Begin token id.
	end_token_id: End token id.
	name: Scope of the variables.
	Returns:
	ids: Output indices.
	logprobs: Output log probabilities probabilities.
	"""
 
	# state = tf.tile(tf.expand_dims(initial_state, axis=1), [1, beam_width, 1])

	sel_sum_logprobs = tf.log([[1.] + [0.] * (beam_width - 1)])
	# ids = tf.tile([[begin_token_id]], [batch_size, beam_width])
	trg_ids = tf.TensorArray(dtype=tf.int32,size=0,dynamic_size=True,clear_after_read=False)
	trg_ids = trg_ids.write(0,tf.fill([batch_size*beam_width],begin_token_id))
	sel_ids = tf.zeros([batch_size, beam_width, 0], dtype=tf.int32)
	mask = tf.ones([batch_size, beam_width], dtype=tf.float32)
	for i in range(sequence_length):
		with tf.variable_scope(name, reuse=True if i > 0 else None):
			trg_input = trg_ids.read(i)
			trg_emb = tf.nn.embedding_lookup(embedding,trg_input)
			dec_outputs,state = update_fn(state=state,inputs=trg_emb)
			# logits = tf.nn.log_softmax(logits)
			# output = tf.reshape(dec_outputs,[-1,self.params.HIDDEN_SIZE*2])
			logits = tf.matmul(dec_outputs,weight) + bias
			# next_id = tf.argmax(logits,axis=1,output_type=tf.int32)
			# trg_ids = trg_ids.write(i+1,next_id)
			# return next_state,trg_ids,step+1
			sum_logprobs = (
			  tf.expand_dims(sel_sum_logprobs, axis=2) +
			  (logits * tf.expand_dims(mask, axis=2)))
			num_classes = logits.shape.as_list()[-1]
			sel_sum_logprobs, indices = tf.nn.top_k(
			  tf.reshape(sum_logprobs, [batch_size, num_classes * beam_width]),
			  k=beam_width)
			ids = indices % num_classes
			trg_ids = trg_ids.write(i+1,tf.reshape(ids,[-1]))
			beam_ids = indices // num_classes

			state = batch_gather(state, beam_ids)
			sel_ids = tf.concat([batch_gather(sel_ids, beam_ids),
							   tf.expand_dims(ids, axis=2)], axis=2)
			mask = (batch_gather(mask, beam_ids) *
				  tf.to_float(tf.not_equal(ids, end_token_id)))

	return sel_ids, sel_sum_logprobs

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
			
	def forward(self,src_input,src_size):#beam search
		batch_size = tf.shape(src_input)[0]
		src_emb = tf.nn.embedding_lookup(self.src_pretrained_embedding,src_input)
		print("src_emb:",src_emb.get_shape().as_list())
		with tf.variable_scope("encoder"):
			enc_outputs,enc_state = tf.nn.bidirectional_dynamic_rnn(
				self.enc_cell_fw,self.enc_cell_bw,src_emb,src_size,dtype=tf.float32)
			enc_outputs = tf.concat([enc_outputs[0],enc_outputs[1]],-1)
			tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(tf.concat([enc_state[0][idx].c,enc_state[1][idx].c],-1),\
				tf.concat([enc_state[0][idx].h,enc_state[1][idx].h],-1)) for idx in range(self.params.NUM_LAYERS)])

			encoder_outputs = tf.contrib.seq2seq.tile_batch(enc_outputs,multiplier=self.params.BEAM_SIZE)
			encoder_state = tf.contrib.framework.nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.params.BEAM_SIZE), tuple_state)
			encoder_inputs_length = tf.contrib.seq2seq.tile_batch(src_size, multiplier=self.params.BEAM_SIZE)
		with tf.variable_scope("decoder"):
			self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.params.HIDDEN_SIZE*2,\
				encoder_outputs,memory_sequence_length=encoder_inputs_length)
		with tf.variable_scope("decoder/rnn/attention_wrapper"):
			self.attention_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_cell,self.attention_mechanism,\
				attention_layer_size=self.params.HIDDEN_SIZE*2)
			state = self.attention_cell.zero_state(batch_size=batch_size*self.params.BEAM_SIZE,dtype=tf.float32).clone(cell_state=encoder_state)
			print("use beam search decoding...")
			sel_ids, sel_sum_logprobs = rnn_beam_search(batch_size,self.tgt_pretrained_embedding,self.softmax_weight,self.softmax_bias,\
				self.attention_cell.call, state, self.params.MAX_DEC_LEN-1, self.params.BEAM_SIZE,
					self.params.SOS_ID, self.params.EOS_ID, name="rnn")
		#   start_tokens = tf.ones([batch_size, ], tf.int32) * self.params.SOS_ID
		#   end_token = self.params.EOS_ID
		# # with tf.variable_scope("decoder/attention_wrapper"):
		#   inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=self.attention_cell, embedding=self.tgt_pretrained_embedding,
		#                                                                        start_tokens=start_tokens, end_token=end_token,
		#                                                                        initial_state=state,
		#                                                                        beam_width=self.params.BEAM_SIZE,
		#                                                                        output_layer=None)
		#   # decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
			#                                                       maximum_iterations=self.params.MAX_DEC_LEN-1)
			# 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，
			# 对于不使用beam_search的时候，它里面包含两项(rnn_outputs, sample_id)
			# rnn_output: [batch_size, decoder_targets_length, vocab_size]
			# sample_id: [batch_size, decoder_targets_length], tf.int32

			# 对于使用beam_search的时候，它里面包含两项(predicted_ids, beam_search_decoder_output)
			# predicted_ids: [batch_size, decoder_targets_length, beam_size],保存输出结果
			# beam_search_decoder_output: BeamSearchDecoderOutput instance namedtuple(scores, predicted_ids, parent_ids)
			# 所以对应只需要返回predicted_ids或者sample_id即可翻译成最终的结果
			# dec_outputs = decoder_outputs.beam_search_decoder_output.scores
			# output = tf.reshape(dec_outputs,[-1,self.params.HIDDEN_SIZE*2])
			# logits = tf.matmul(output,self.softmax_weight) + self.softmax_bias

			# ids = tf.argmax(tf.reshape(logits,[-1,self.params.TGT_VOCAB_SIZE]),axis=-1,output_type=tf.int32)
			# ids = tf.reshape(ids,[-1,self.params.BEAM_SIZE])
		return sel_ids
