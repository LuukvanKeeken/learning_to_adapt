import tensorflow as tf


a = tf.placeholder(dtype=tf.int32)
b = tf.placeholder(dtype=tf.int32)
c = tf.constant(5)
d = tf.add(b, c)
e = tf.add(a, c)


with tf.Session() as sess:
    print(sess.run([d, e], feed_dict = {b: 8}))