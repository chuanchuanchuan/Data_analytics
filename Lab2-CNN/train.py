import os
import tensorflow as tf
import forward
#加载forward.py中定义的常量和前向传播的函数
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import pickle
import mnist_reader
BATCH_SIZE = 10000
LEARNING_RATE = 0.8
TRAINING_STEPS = 30000

#模型保存的路径和文件名
MODEL_SAVE_PATH = "model"
MODEL_NAME = "model.ckpt"

def train(data_set):
	#定义输入输出placeholder
	x = tf.placeholder(
	tf.float32,[None,forward.CONV1_SIZE,forward.CONV1_SIZE,forward.CHANNELS],name='x-input')
	y_ = tf.placeholder(
	tf.float32,[None,forward.LABELS],name ='y-input')
	
	#直接使用forward.py中定义的前向传播过程
	
	y = forward.forward(x,train=True)
	
	global_step = tf.Variable(0,trainable=False)
	
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
	y,tf.argmax(y_,1))
	
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
	
	train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE)\
					minimize(loss,global_step=global_step)
	
	#初始化Tensorflow持久化类
	#data = input_data.read_data_sets('data/fashion'(data_set))
	x_list = []
	y_list = []
	for i in range(5):
		data_batch = "data_batch_" + str(i+1)
		with open(data_set+data_batch,'rb') as fo:
			dict = pickle.load(fo,encoding='bytes')
			x_list.append(dict[b'data'])
			y_list.append(dict[b'labels'])

	"""
	x_list = []
	y_list = []
	x_train,y_train = mnist_reader.load_mnist('data/fashion',kind = 'train')
	for i in range(6):
		x_list.append(x_train[i*10000:(i+1)*10000])
		y_list.append(y_train[i*10000:(i+1)*10000])
	"""
	saver = tf.train.Saver()
	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		
		for i in range(TRAINING_STEPS):
			#得到每一个batch的数据 xs ys
			xs = x_list[i % 5]
			ys = y_list[i % 5]
			"""
			xs = x_list[i % 6]
			ys = y_list[i % 6]
			"""
			reshape_xs = np.reshape(xs,(BATCH_SIZE,forward.IMAGE_SIZE,forward.IMAGE_SIZE,forward.CHANNELS))
			
			_, loss_value, step = sess.run([train_step,loss,global_step],
											feed_dict = {x:reshape_xs,y_:ys})
			
			#每1000轮保存一轮模型
			
			if i % 1000 == 0:
				print("After %d training step(s), loss on training "
				      "batch is %g." % (step,loss_value))
				
				saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)


train("cifar-10-python\cifar-10-batches-py\")			