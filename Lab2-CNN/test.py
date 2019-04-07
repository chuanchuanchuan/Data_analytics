import forward.py
import tensorflow as tf
import numpy as np

def test(data_set,model_path):
	#定义输入输出placeholder
	x = tf.placeholder(
	tf.float32,[None,forward.CONV1_SIZE,forward.CONV1_SIZE,forward.CHANNELS],name='x-input')
	y_ = tf.placeholder(
	tf.float32,[None,forward.LABELS],name ='y-input')
	
	#读取数据
	with open("cifar-10-python\cifar-10-batches-py\(data_set)test_batch",'rb') as fo:
			dict = pickle.load(fo,encoding='bytes')
	validate_feed = {x:dict['data'],y_:dict['labels']}
	#data = input_data.read_data_sets("data_set")
	#validate_feed = {x:data.validation.images,y_:data.validation.labels}
	y = forward.forward(x,train=False)
	
	correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	
	sess = tf.InteractiveSession()
	
	saver = tf.train.saver()
	saver.restore(sess,model_path)
	
	accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
	print("accuracy = %f" % (accuracy_score))
	
	