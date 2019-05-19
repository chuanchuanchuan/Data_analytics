import forward
import train
import tensorflow as tf
import numpy as np
import mnist_reader
import pickle

def test(data_set, model_path):
    # 定义输入输出placeholder
    x = tf.placeholder(
        tf.float32, [None, forward.IMAGE_SIZE, forward.IMAGE_SIZE, forward.CHANNELS], name='x-input')
    y_ = tf.placeholder(
        tf.float32, None, name='y-input')

    # 读取数据
    """
    with open(data_set, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    """
    list1 = []
    for i in range(10):
        """
        xs = np.reshape(dict[b'data'][i*1000:(i+1)*1000], (1000, forward.IMAGE_SIZE, forward.IMAGE_SIZE, forward.CHANNELS),order='F')
        validate_feed = {x: xs, y_: dict[b'labels'][i*1000:(i+1)*1000]}
        """
        x_test,y_test = mnist_reader.load_mnist(data_set,kind='t10k')
        xs = np.reshape(x_test[i*1000:(i+1)*1000],(1000,forward.IMAGE_SIZE,forward.IMAGE_SIZE,forward.CHANNELS),order='F')
        validate_feed = {x:xs/255,y_:y_test[i*1000:(i+1)*1000]}
        y = forward.forward(x, train=False)

        correct_prediction = tf.equal(tf.to_int64(tf.argmax(y, 1)), tf.to_int64(y_))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess = tf.InteractiveSession()

        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
        print(accuracy_score)
        list1.append(accuracy_score)
    print("accuracy = %g" % (int(sum(list1))/int(10)))


test('data/fashion',"model/model.ckpt-3001")