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

	def forward(self,src_input,src_size,trg_input,trg_label,trg_size):
		batch_size = tf.shape(src_input)[0]
		src_emb = tf.nn.embedding_lookup(self.src_pretrained_embedding,src_input)
		trg_emb = tf.nn.embedding_lookup(self.tgt_pretrained_embedding,trg_input)
		src_emb = tf.nn.dropout(src_emb,self.params.KEEP_PROB)
		trg_emb = tf.nn.dropout(trg_emb,self.params.KEEP_PROB)

		with tf.variable_scope("encoder",reuse=tf.AUTO_REUSE):
			enc_outputs,enc_state = tf.nn.bidirectional_dynamic_rnn(
				self.enc_cell_fw,self.enc_cell_bw,src_emb,src_size,dtype=tf.float32)
			enc_outputs = tf.concat([enc_outputs[0],enc_outputs[1]],-1)
			tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(tf.concat([enc_state[0][idx].c,enc_state[1][idx].c],-1),\
			tf.concat([enc_state[0][idx].h,enc_state[1][idx].h],-1)) for idx in range(self.params.NUM_LAYERS)])


		with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE):
			self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.params.HIDDEN_SIZE*2,\
				enc_outputs,memory_sequence_length=src_size)
			self.attention_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_cell,self.attention_mechanism,\
				attention_layer_size=self.params.HIDDEN_SIZE*2)
			state = self.attention_cell.zero_state(batch_size=batch_size,dtype=tf.float32).clone(cell_state=tuple_state)
			dec_outputs,_ = tf.nn.dynamic_rnn(
				self.attention_cell,trg_emb,trg_size,initial_state=state,dtype=tf.float32)

		output = tf.reshape(dec_outputs,[-1,self.params.HIDDEN_SIZE*2])
		logits = tf.matmul(output,self.softmax_weight) + self.softmax_bias
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_label,[-1]),logits=logits)

		label_weights = tf.sequence_mask(trg_size,maxlen=tf.shape(trg_label)[1],dtype=tf.float32)
		label_weights = tf.reshape(label_weights,[-1])

		cost = tf.reduce_sum(loss*label_weights)
		cost_per_token = cost / tf.reduce_sum(label_weights)
		

		trainable_variables = tf.trainable_variables()
		grads = tf.gradients(cost / tf.to_float(batch_size),trainable_variables)
		grads,_ = tf.clip_by_global_norm(grads,self.params.MAX_GRAD_NORM)

		optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.params.LEARNING_RATE)
		train_op = optimizer.apply_gradients(zip(grads,trainable_variables))

		return cost_per_token,train_op
