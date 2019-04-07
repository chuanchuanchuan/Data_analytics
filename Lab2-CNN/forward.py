# forward fuction 卷积神经网络中的前向过程

import tensorflow as tf

#配置神经网络的参数
#实现类似LeNet-5模型

INPUT_NODE = 3072
OUTPUT_NODE = 10

IMAGE_SIZE = 32
CHANNELS = 3
LABELS = 10

#第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
#第二层卷积层的尺度和深度
CONV2_DEEP = 64
CONV2_SIZE = 5
#全连接层的结点个数
FC_NUM = 512


def forward(input,train,regularizer):
	with tf.variable_scope('layer1-conv1'):
		conv1_weights = tf.get_variable(
		"weight",[CONV1_SIZE,CONV1_SIZE,CHANNELS,CONV1_DEEP],
		initializer = tf.truncate_normal_initializer(stddev=0.1))
		conv1_biases = tf.get_variable(
		"bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
		#输入32*32*3 输出28*28*32
		
	conv1 = tf.nn.con2d(input,conv1_weights,strides=[1,1,1,1],padding = 'VALID')
	relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
	
	with tf.name_scope('layer2-pool1'):
		pool1 = tf.nn.max_pool(
		relu1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
		#输入28*28*32 输出14*14*32
	
	with tf.variable_scope('layer3-conv2'):
		conv2_weights = tf.get_variable(
		"weight",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP，CONV2_DEEP],
		initializer = tf.truncate_normal_initializer(stddev=0.1))
		conv2_biases = tf.get_variable(
		"bias",[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
		#输入14*14*32 输出14*14*64
		
		conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
		
	with tf.name_scope('layer4-pool2'):
		pool2 = tf.nn.max_pool(
		relu2,ksize=[1,2,2,1],strides = [1,2,2,1],padding = 'VALID')
		#输入14*14*64 输出7*7*64
		
	pool_shape = pool2.get_shape().as_list()
	nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
	reshaped = tf.reshaped(pool2,[pool_shape[0],nodes])
	
	with tf.variable_scope('layer5-fc1'):
		fc1_weights = tf.get_variable(
		'weight',[nodes,FC_NUM],initializer=tf.truncate_normal_initializer(stddev=0.1))
		
		if regularizer != None:
			tf.add_to_collection('losses',regularizer(fc1_weights))
		fc1_biases = tf.get_variable(
		"bias",[FC_NUM],initializer=tf.constant_initializer(0.1))
		
		fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
		if train: fc1 = tf.nn.dropout(fc1,0.5) #dropout只在训练模型
		
	with tf.variable_scope('layer6-fc2'):
		fc2_weights = tf.get_variable(
		"weight",[FC_NUM,LABELS],initializer=tf.truncate_normal_initializer(stddev=0.1))
		if regularizer != None:
			tf.add_to_collection('losses',regularizer(fc2_weights))
		fc2_biases = tf.get_variable(
		"bias",[LABELS],initializer=tf.constant_initializer(0.1))
		logit = tf.matmul(fc1,fc2_weights) + fc2_biases
	
	return logit
		
	