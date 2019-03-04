import tensorflow as tf
from config import opt

def MakeDataset(file_path):
	dataset = tf.data.TextLineDataset(file_path)
	dataset = dataset.map(lambda string:tf.string_split([string]).values)#split
	dataset = dataset.map(lambda string:tf.string_to_number(string,tf.int32))#int
	dataset = dataset.map(lambda x:(x,tf.size(x)))#add length
	return dataset

def MakeSrcTrgDataset(src_path,trg_path,batch_size,isTrain=True):
	with tf.name_scope("src_data"):
		src_data = MakeDataset(src_path)
	with tf.name_scope("trg_data"):	
		trg_data = MakeDataset(trg_path)

	dataset = tf.data.Dataset.zip((src_data,trg_data))

	def FilterLength(src_tuple,trg_tuple):
		((src_input,src_len),(trg_label,trg_len)) = (src_tuple,trg_tuple)
		src_len_ok = tf.logical_and(tf.greater(src_len,1),tf.less_equal(src_len,opt.MAX_LEN_S))
		trg_len_ok = tf.logical_and(tf.greater(trg_len,1),tf.less_equal(trg_len,opt.MAX_LEN_Q))
		return tf.logical_and(src_len_ok,trg_len_ok)

	dataset = dataset.filter(FilterLength)

	def MakeTrgInput(src_tuple,trg_tuple):
		((src_input,src_len),(trg_label,trg_len)) = (src_tuple,trg_tuple)
		trg_input = tf.concat([[opt.SOS_ID],trg_label[:-1]],axis=0)
		return ((src_input,src_len),(trg_input,trg_label,trg_len))

	dataset = dataset.map(MakeTrgInput)

	if isTrain:
		dataset = dataset.shuffle(10000)
		
	padded_shapes = ((tf.TensorShape([None]),tf.TensorShape([])),
		(tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([])))
	batched_dataset = dataset.padded_batch(batch_size,padded_shapes)
	return batched_dataset
