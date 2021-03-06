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

	def index_matrix_to_pairs(self,index_matrix):
		# [[3,1,2], [2,3,1]] -> [[[0, 3], [1, 1], [2, 2]], 
		#                        [[0, 2], [1, 3], [2, 1]]]
		replicated_first_indices = tf.range(tf.shape(index_matrix)[0])#batch_size
		rank = len(index_matrix.get_shape())
		if rank == 2:
			replicated_first_indices = tf.tile(
				tf.expand_dims(replicated_first_indices, dim=1),
				[1, tf.shape(index_matrix)[1]])
		return tf.stack([replicated_first_indices, index_matrix], axis=rank)

	def attention(self,ref,query,with_softmax,scope="attention"):
		with tf.variable_scope(scope):
			W_ref = tf.get_variable("W_ref",[1,hidden_dim,hidden_dim],initializer=initializer)
			W_q = tf.get_variable("W_q",[hidden_dim,hidden_dim],initializer=initializer)
			v = tf.get_variable("v",[hidden_dim],initializer=initializer)

			encoded_ref = tf.nn.conv1d(ref,W_ref,1,"VALID",name="encoded_ref")
			encoded_query = tf.expand_dims(tf.matmul(query,W_q,name="encoded_query"),1)
			tiled_encoded_query = tf.tile(encoded_query,[1,tf.shape(encoded_ref)[1],1],name="tiled_encoded_query")
			scores = tf.reduce_sum(v*tf.tanh(encoded_ref+encoded_query),[-1])

			if with_softmax:
				return tf.nn.softmax(scores)
			else:
				return scores

	def glimpse(self,ref,query,scope="glimpse"):
		p = attention(ref,query,with_softmax=True,scope=scope)
		alignments = tf.expand_dims(p,2)
		return tf.reduce_sum(alignments*ref,[1])

		
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

		with tf.variable_scope("encoder"):
			self.enc_cell_fw = self._create_rnn_cell(self.params.HIDDEN_SIZE)
			self.enc_cell_bw = self._create_rnn_cell(self.params.HIDDEN_SIZE)
			enc_outputs,enc_state = tf.nn.bidirectional_dynamic_rnn(
				self.enc_cell_fw,self.enc_cell_bw,src_emb,self.src_size,dtype=tf.float32)
			enc_outputs = tf.concat([enc_outputs[0],enc_outputs[1]],-1)

			#self attention
			Ws = tf.get_variable(name='Ws',shape=[self.params.HIDDEN_SIZE*2,self.params.HIDDEN_SIZE*2],initializer=tf.truncated_normal_initializer(stddev=0.1))
			U = tf.identity(enc_outputs)
			A = tf.nn.softmax(tf.batch_matmul(tf.matmul(U,Ws),tf.transpose(enc_outputs,perm=[2,1])),axis=-1)
			S = tf.batch_matmul(A,U)
			#feature fusion gating
			Wf = tf.get_variable(name='Wf',shape=[self.params.HIDDEN_SIZE*4,self.params.HIDDEN_SIZE*2],initializer=tf.truncated_normal_initializer(stddev=0.1))
			F = tf.tanh(tf.matmul(tf.concat([U,S],-1),Wf))
			Wg = tf.get_variable(name='Wg',shape=[self.params.HIDDEN_SIZE*4,self.params.HIDDEN_SIZE*2],initializer=tf.truncated_normal_initializer(stddev=0.1))
			G = tf.sigmoid(tf.matmul(tf.concat([U,S],-1),Wg))
			U_ = G*F + (1-G)*U

			tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(tf.concat([enc_state[0][idx].c,enc_state[1][idx].c],-1),\
			tf.concat([enc_state[0][idx].h,enc_state[1][idx].h],-1)) for idx in range(self.params.NUM_LAYERS)])

		with tf.variable_scope("decoder_scope"):
			src_size = self.src_size
			batch_size = self.batch_size
			if self.params.MODE == 'test':
				print("use beamsearch decoding..")
				print("BEAM_SIZE is %d!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"%(self.params.BEAM_SIZE))
				U_ = tf.contrib.seq2seq.tile_batch(U_, multiplier=self.params.BEAM_SIZE)
				tuple_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.params.BEAM_SIZE), tuple_state)
				src_size = tf.contrib.seq2seq.tile_batch(self.src_size, multiplier=self.params.BEAM_SIZE)
				batch_size = self.batch_size * self.params.BEAM_SIZE

			self.dec_cell = self._create_rnn_cell(self.params.HIDDEN_SIZE*2)
			self.attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.params.HIDDEN_SIZE*2,\
				U_,memory_sequence_length=src_size,name='LuongAttention')
			self.attention_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_cell,self.attention_mechanism,\
				attention_layer_size=self.params.HIDDEN_SIZE*2,alignment_history=True,name='Attention_Wrapper')
			decoder_initial_state = self.attention_cell.zero_state(batch_size=batch_size,dtype=tf.float32).clone(cell_state=tuple_state)
		
			self.output_layer = tf.layers.Dense(self.params.TGT_VOCAB_SIZE, \
				kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

			# if self.params.MODE in ['train','val']:
			# 	trg_emb = tf.nn.embedding_lookup(self.tgt_pretrained_embedding,self.trg_input)
			# 	print(trg_emb.get_shape().as_list())
				
			# 	training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=trg_emb,
			# 														sequence_length=self.trg_size,
			# 														time_major=False, name='training_helper')
			# 	training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.attention_cell, helper=training_helper,
			# 													   initial_state=decoder_initial_state, output_layer=self.output_layer)
			# 	# decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
			# 	# 														  impute_finished=True,
			# 	# 													maximum_iterations=tf.shape(self.trg_label)[1],scope="decoder")
			# 	(training_finished, training_first_inputs, training_initial_state) = training_decoder.initialize(name="decoder")
			# 	# self.training_outputs_list = []
			# 	# self.training_alignments_list = []
			# 	self.final_distributions = []
			# 	for t in range(tf.shape(self.trg_label)[1]):
			# 		if t>0: tf.get_variable_scope().reuse_variables()
			# 		(training_outputs, training_next_state, training_next_inputs, training_finished) = \
			# 			training_decoder.step(t,training_first_inputs,training_initial_state)
			# 		self.final_distributions.append(tf.nn.softmax(tf.concat([tf.nn.softmax(training_outputs,axis=-1),training_next_state.alignments],-1),axis=-1))
				
			# 	rnn_output = tf.concat(self.final_distributions,-1)
			# 	self.decoder_logits_train = tf.identity(rnn_output)
				
			# 	# self.mask = tf.sequence_mask(self.trg_size, maxlen=tf.shape(self.trg_label)[1], dtype=tf.float32, name='masks')
			# 	self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,targets=self.trg_label, weights=self.mask)
			# 	self.summary = tf.summary.scalar('loss', self.loss)
			# 	optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.params.LEARNING_RATE)
			# 	trainable_params = tf.trainable_variables()
			# 	gradients = tf.gradients(self.loss, trainable_params)
			# 	clip_gradients, _ = tf.clip_by_global_norm(gradients, self.params.MAX_GRAD_NORM)
			# 	self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
			# 	print('building training stage model finished ... ...')

			# elif self.params.MODE == 'test':
			# 	start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.params.SOS_ID
			# 	end_token = self.params.EOS_ID
			# 	inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=self.attention_cell, embedding=self.tgt_pretrained_embedding,
			# 															 start_tokens=start_tokens, end_token=end_token,
			# 															 initial_state=decoder_initial_state,
			# 															 beam_width=self.params.BEAM_SIZE,
			# 															 output_layer=self.output_layer)
			# 	decoder_outputs,final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
			# 													maximum_iterations=self.params.MAX_DEC_LEN-1,scope="decoder")
			# 	self.alignment_history = final_state.cell_state.alignment_history
			# 	self.decoder_predict_decode = decoder_outputs.predicted_ids
			# 	print('building testing stage model finished ... ...')