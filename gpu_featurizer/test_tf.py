import time
import numpy as np
import tensorflow as tf

ani_mod = tf.load_op_library('ani.so');

Xs = np.load('Xs.npy')
Ys = np.load('Ys.npy')
Zs = np.load('Zs.npy')
As = np.load('As.npy')
MOs = np.load('MOs.npy')
MACs = np.load('MACs.npy')
# TCs = np.zeros(4, dtype=np.int32)

# for z in As:
# 	# print(z)
# 	TCs[z] += 1

feat = ani_mod.ani(Xs, Ys, Zs, As, MOs, MACs)

st = time.time()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

	for idx in range(1000):
		results = sess.run(feat)
		print(len(results))
		print("python samples per minute: ", (idx+1)*len(MOs)/(time.time()-st) * 60)

	# res = res.reshape(len(Xs), 384)
	# for f in res:
		# print(f)

	# print(res.shape)