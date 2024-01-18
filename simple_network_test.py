import numpy as np
import tensorflow as tf
import time

start_time = time.time()
# One 3D tensor and one 2D tensor:
x = tf.constant(np.random.random((100, 16, 5)))
param = tf.constant(np.random.random((5, 512)))
result = tf.tensordot(x, param, axes=[[2], [0]])

with tf.Session() as sess:
    for i in range(100000):
        sess.run(result)

total_time = time.time() - start_time

print("Total time: {}".format(total_time))
print("Time per iteration: {}".format(total_time / 100000))


start_time = time.time()
x_reshaped = tf.reshape(x, (-1, 5))
result = tf.matmul(x_reshaped, param)
result = tf.reshape(result, (100, 16, 512))

with tf.Session() as sess:
    for i in range(100000):
        sess.run(result)

total_time = time.time() - start_time

print("Total time: {}".format(total_time))
print("Time per iteration: {}".format(total_time / 100000))