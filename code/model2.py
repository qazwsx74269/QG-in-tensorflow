#coding:utf-8
import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest

class QGModel(object):
	def __init__(self,params):
		self.params = params
		self.src_pretrained_embedding = tf.convert_to_tensor(np.load(self.params.SRC_ID2EMBEDDING),dtype=tf.float32)
		self.tgt_pretrained_embedding = tf.convert_to_tensor(np.load(self.params.TGT_ID2EMBEDDING),dtype=tf.float32)
		self.build_model()

	def _create_rnn_cell(self,size):
		def single_rnn_cell(size):
			single_cell = tf.contrib.rnn.LSTMCell(size)
			cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder)
			return cell
		cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell(size) for _ in range(self.params.NUM_LAYERS)])
		return cell

	def build_model(self):
		print('building model... ...')
		self.src_input = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
		self.src_size = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')
		self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
		self.trg_input = tf.placeholder(tf.int32, [None, None], name='decoder_inputs')
		self.trg_label = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
		self.trg_size = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
		self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')

		src_emb = tf.nn.embedding_lookup(self.src_pretrained_embedding,self.src_input)
		# src_emb = tf.nn.dropout(src_emb,self.params.KEEP_PROB)

		with tf.variable_scope("encoder"):
			# self.enc_cell_fw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.params.HIDDEN_SIZE)\
			#   for _ in range(self.params.NUM_LAYERS)])
			# self.enc_cell_bw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.params.HIDDEN_SIZE)\
			#   for _ in range(self.params.NUM_LAYERS)])
			self.enc_cell_fw = self._create_rnn_cell(self.params.HIDDEN_SIZE)
			self.enc_cell_bw = self._create_rnn_cell(self.params.HIDDEN_SIZE)
			enc_outputs,enc_state = tf.nn.bidirectional_dynamic_rnn(
				self.enc_cell_fw,self.enc_cell_bw,src_emb,self.src_size,dtype=tf.float32)
			enc_outputs = tf.concat([enc_outputs[0],enc_outputs[1]],-1)
			tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(tf.concat([enc_state[0][idx].c,enc_state[1][idx].c],-1),\
			tf.concat([enc_state[0][idx].h,enc_state[1][idx].h],-1)) for idx in range(self.params.NUM_LAYERS)])

		with tf.variable_scope("decoder_scope"):
			src_size = self.src_size
			if self.params.BEAM_SEARCH and self.params.MODE == 'test':
				# 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制params.BEAM_SIZE份。
				print("use beamsearch decoding..")
				enc_outputs = tf.contrib.seq2seq.tile_batch(enc_outputs, multiplier=self.params.BEAM_SIZE)
				tuple_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.params.BEAM_SIZE), tuple_state)
				src_size = tf.contrib.seq2seq.tile_batch(self.src_size, multiplier=self.params.BEAM_SIZE)
			# self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.params.HIDDEN_SIZE*2)\
			#   for _ in range(self.params.NUM_LAYERS)])
			self.dec_cell = self._create_rnn_cell(self.params.HIDDEN_SIZE*2)
			self.attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.params.HIDDEN_SIZE*2,\
				enc_outputs,memory_sequence_length=src_size,name='LuongAttention')
			self.attention_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_cell,self.attention_mechanism,\
				attention_layer_size=self.params.HIDDEN_SIZE*2,alignment_history=True,name='Attention_Wrapper')
			batch_size = self.batch_size if not self.params.BEAM_SEARCH else self.batch_size * self.params.BEAM_SIZE
			decoder_initial_state = self.attention_cell.zero_state(batch_size=batch_size,dtype=tf.float32).clone(cell_state=tuple_state)
		
			self.output_layer = tf.layers.Dense(self.params.TGT_VOCAB_SIZE, \
				kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

			if self.params.MODE in ['train','val']:
				trg_emb = tf.nn.embedding_lookup(self.tgt_pretrained_embedding,self.trg_input)
				# trg_emb = tf.nn.dropout(trg_emb,self.params.KEEP_PROB)
				print(trg_emb.get_shape().as_list())
				
				training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=trg_emb,
																	sequence_length=self.trg_size,
																	time_major=False, name='training_helper')
				training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.attention_cell, helper=training_helper,
																   initial_state=decoder_initial_state, output_layer=self.output_layer)
				decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
																		  impute_finished=True,
																	maximum_iterations=tf.shape(self.trg_label)[1],scope="decoder")
				# with tf.variable_scope("run_decoder", reuse=tf.AUTO_REUSE):
    			#	for i in range(tf.shape(self.trg_label)[1]):
    			#		cell_output, state=training_decoder.step(i,decoder_inputs, state)
    			#		print(cell_output.get_shape().as_list(),state.get_shape().as_list())

				self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
				# self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train')
				self.mask = tf.sequence_mask(self.trg_size, maxlen=tf.shape(self.trg_label)[1], dtype=tf.float32, name='masks')
				self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,targets=self.trg_label, weights=self.mask)
				self.summary = tf.summary.scalar('loss', self.loss)
				optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.params.LEARNING_RATE)
				trainable_params = tf.trainable_variables()
				gradients = tf.gradients(self.loss, trainable_params)
				clip_gradients, _ = tf.clip_by_global_norm(gradients, self.params.MAX_GRAD_NORM)
				self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
				print('building training stage model finished ... ...')

				# return self.loss,self.train_op  

			elif self.params.MODE == 'test':
				start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.params.SOS_ID
				end_token = self.params.EOS_ID
				# decoder阶段根据是否使用beam_search决定不同的组合，
				# 如果使用则直接调用BeamSearchDecoder（里面已经实现了helper类）
				# 如果不使用则调用GreedyEmbeddingHelper+BasicDecoder的组合进行贪婪式解码
				if self.params.BEAM_SEARCH:
					inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=self.attention_cell, embedding=self.tgt_pretrained_embedding,
																			 start_tokens=start_tokens, end_token=end_token,
																			 initial_state=decoder_initial_state,
																			 beam_width=self.params.BEAM_SIZE,
																			 output_layer=self.output_layer)
				else:
					decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.tgt_pretrained_embedding,
																			   start_tokens=start_tokens, end_token=end_token)
					inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.attention_cell, helper=decoding_helper,
																		initial_state=decoder_initial_state,
																		output_layer=self.output_layer)
				decoder_outputs,final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
																maximum_iterations=self.params.MAX_DEC_LEN-1,scope="decoder")
				if self.params.BEAM_SEARCH:
					self.alignment_history = final_state.cell_state.alignment_history
				else:
					self.alignment_history = final_state.alignment_history#有错误

				# 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，
				# 对于不使用beam_search的时候，它里面包含两项(rnn_outputs, sample_id)
				# rnn_output: [batch_size, decoder_targets_length, vocab_size]
				# sample_id: [batch_size, decoder_targets_length], tf.int32

				# 对于使用beam_search的时候，它里面包含两项(predicted_ids, beam_search_decoder_output)
				# predicted_ids: [batch_size, decoder_targets_length, params.BEAM_SIZE],保存输出结果
				# beam_search_decoder_output: BeamSearchDecoderOutput instance namedtuple(scores, predicted_ids, parent_ids)
				# 所以对应只需要返回predicted_ids或者sample_id即可翻译成最终的结果
				if self.params.BEAM_SEARCH:
					self.decoder_predict_decode = decoder_outputs.predicted_ids
				else:
					self.decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)
				print('building testing stage model finished ... ...')

			# return self.decoder_predict_decode

