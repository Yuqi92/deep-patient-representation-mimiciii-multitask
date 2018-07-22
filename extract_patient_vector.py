import tensorflow as tf
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from utility import generate_token_embedding, CNN_model
import logging
import HP
from multiprocessing import Pool
from Embedding import Embedding

_ = Embedding.get_embedding()

logging.basicConfig(filename=HP.log_file_name, level=logging.INFO)

result_csv = pd.read_csv(HP.result_csv)
patient_ids = np.asarray(result_csv["patient_id"])
n_patient = len(patient_ids)

# define placeholders
input_x = tf.placeholder(tf.float32,
                         [None, HP.max_document_length, HP.max_sentence_length, HP.embedding_size],
                         name="input_x")
sent_length = tf.placeholder(tf.int32, [None], name="sent_length")
input_ys = []
for i in range(HP.multi_size):
    input_ys.append(tf.placeholder(tf.int32, [None, HP.num_classes], name="input_y"+str(i)))
category_index = tf.placeholder(tf.int32, [None, HP.max_document_length], name='category_index')
dropout_keep_prob = tf.placeholder(tf.float32, [], name="dropout_keep_prob")
_, _, patient_vector = CNN_model(input_x, input_ys, sent_length, category_index, dropout_keep_prob)

saver = tf.train.Saver()
num_batch = int(math.ceil(n_patient / HP.n_batch))
with tf.Session() as sess:
    saver.restore(sess, HP.model_path)

    # start train
    for i in tqdm(range(num_batch)):
        tmp_train_patient_name = patient_ids[i*HP.n_batch:min((i+1)*HP.n_batch, n_patient)]
        pool = Pool(processes=HP.read_data_thread_num)
        generate_token_embedding_results = pool.map(generate_token_embedding, tmp_train_patient_name)
        pool.close()
        pool.join()

        tmp_x = np.zeros([len(generate_token_embedding_results),
                         HP.n_max_sentence_num,
                         HP.n_max_word_num,
                         HP.embedding_size], dtype=np.float32)
        l = []
        tmp_cate = []
        for (M, r) in enumerate(generate_token_embedding_results):
            tmp_x[M] = r[0]
            l.append(r[1])
            tmp_cate.append(r[2])

        cate_id = np.stack(tmp_cate)
        l = np.asarray(l)
        feed_dict = {input_x: tmp_x,
                     sent_length: l,
                     category_index: cate_id,
                     dropout_keep_prob: 1.0}
        # logging.info("start to train")
        tmp_patient_vector = sess.run(patient_vector, feed_dict=feed_dict)
        for j in range(len(tmp_train_patient_name)):
            np.save(HP.patient_vector_directory + tmp_train_patient_name[i] + ".npy", tmp_patient_vector[j])
