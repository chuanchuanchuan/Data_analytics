import forward.py
import train.py
import tensorflow as tf
import numpy as np
import mnist_reader
def test(data_set,model_path):
	#定义输入输出placeholder
	x = tf.placeholder(
	tf.float32,[None,forward.CONV1_SIZE,forward.CONV1_SIZE,forward.CHANNELS],name='x-input')
	y_ = tf.placeholder(
	tf.float32,[None,forward.LABELS],name ='y-input')
	
	#读取数据
	with open("cifar-10-python\cifar-10-batches-py\(data_set)test_batch",'rb') as fo:
			dict = pickle.load(fo,encoding='bytes')
	xs = np.shape(dict[b'data'],(train.BATCH_SIZE,forward.IMAGE_SIZE,forward.IMAGE_SIZE,forward.CHANNELS))
	validate_feed = {x:xs,y_:dict[b'labels']}
	#x_test,y_test = mnist_reader.load_minst('data/fashion',kind='t10k')
	#xs = np.shape(x_test,(train.BATCH_SIZE,forward.IMAGE_SIZE,forward.IMAGE_SIZE,forward.CHANNELS))
	#validate_feed = {x:xs,y_:y_test}
	y = forward.forward(x,train=False)
	
	correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	
	sess = tf.InteractiveSession()
	
	saver = tf.train.saver()
	saver.restore(sess,model_path)
	
	accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
	print("accuracy = %g" % (accuracy_score))
	
	