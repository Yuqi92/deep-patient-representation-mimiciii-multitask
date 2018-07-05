import os
import glob
import tensorflow as tf
import numpy as np
import math
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from utility import  extract_embedding, generate_token_embedding, split_train_test_dev, CNN_model,generate_sent_category

import logging
logging.basicConfig(filename="example.log", level=logging.INFO)



# input file and note
logging.info('build data:')
data_folder_path = '../CNN_mimic_iii/dead_in_month/file/truncated_files_with_category/'
pos_path = data_folder_path + 'pos/'
neg_path = data_folder_path + 'neg/'
pos_files = glob.glob(pos_path + "*.txt")
neg_files = glob.glob(neg_path + "*.txt")

# split train test dev
logging.info('split_train_test_dev')
load_path = '.../dead_in_month/index/'
train_file,test_file,dev_file,y_train,y_test,y_dev = split_train_test_dev(pos_files, neg_files, load_path, False)

# generate mimic embedding
logging.info('extract mimic')
embedding_folder = '.../data/glove.6B'
file_embedding= open(os.path.join(embedding_folder,'mimic.k100.w2v'))
mimic3_embedding = extract_embedding(file_embedding)

# train CNN model

embedding_size = 100
max_document_length = 1000
max_sentence_length = 25
n_class = 2
n_batch = 64
early_stop_times = 5
num_train_batch = math.ceil(len(train_file) / n_batch)
num_dev_batch = math.ceil(len(dev_file) / n_batch)
num_test_batch = math.ceil(len(test_file) / n_batch)

input_x = tf.placeholder(tf.float32, [None, max_document_length, max_sentence_length,embedding_size], name="input_x")
input_y = tf.placeholder(tf.int32, [None, n_class], name="input_y")
sent_length = tf.placeholder(tf.int32, [None], name="sent_length")

# category placeholder
category_index = tf.placeholder(tf.int32, [None, max_document_length], name='category_index')
dropout_keep_prob = tf.placeholder(tf.float32, [],name="dropout_keep_prob")
#lr_placeholder = tf.placeholder(tf.float32, [],name="lr")

optimize, predictions = CNN_model(input_x, input_y, sent_length, category_index, dropout_keep_prob)
saver = tf.train.Saver()

with tf.Session() as sess:
    restore = False
    if restore:
        saver.restore(sess, "dead_in_month/results/model_1/model.weights/model.ckpt")
    else:
        sess.run(tf.global_variables_initializer())
    shuf_ind = np.asarray(list(range(len(train_file))))
    max_auc = 0
    current_early_stop_times = 0

    while True:
        np.random.shuffle(shuf_ind)
        train_file = train_file[shuf_ind]
        y_train = y_train[shuf_ind]
        #lr_decay = 0.99
        #learning_rate = 0.001

        # start train
        for i in tqdm(range(num_train_batch)):
            tmp_train_file_name_list = train_file[i*n_batch:min((i+1)*n_batch, len(train_file))]
            tmp_y_train = y_train[i*n_batch:min((i+1)*n_batch, len(train_file))]
            tmp_x_train = []
            l = []
            tmp_cate = []
            for f in tmp_train_file_name_list:
                new_x_train, new_l, new_cate = generate_token_embedding(f, mimic3_embedding)
                tmp_x_train.append(new_x_train)
                l.append(new_l)
                tmp_cate.append(new_cate)
            tmp_x_train = np.stack(tmp_x_train)
            cate_id = np.stack(tmp_cate)
            l = np.asarray(l)
            sess.run([optimize],
                     feed_dict={
                         input_x: tmp_x_train,
                         input_y: tmp_y_train,
                         sent_length: l,
                         category_index: cate_id,
                         dropout_keep_prob: 0.8
                     })


        # get validation result
        y_dev_label = []
        predictions_dev = []

        for i in range(num_dev_batch):
            tmp_dev_file_name_list = dev_file[i*n_batch:min((i+1)*n_batch, len(dev_file))]
            tmp_y_dev = y_dev[i*n_batch:min((i+1)*n_batch, len(dev_file))]
            y_dev_label.extend(np.argmax(tmp_y_dev,axis=1).tolist())

            tmp_x_dev = []
            l = []
            tmp_cate = []
            for f in tmp_dev_file_name_list:
                new_x_dev, new_l, new_cate = generate_token_embedding(f, mimic3_embedding)
                tmp_x_dev.append(new_x_dev)
                l.append(new_l)
                tmp_cate.append(new_cate)
            tmp_x_dev = np.stack(tmp_x_dev)
            cate_id = np.stack(tmp_cate)
            l = np.asarray(l)

            pre = sess.run(predictions,
                feed_dict=
                {input_x: tmp_x_dev,
                input_y: tmp_y_dev,
                sent_length: l,
                category_index: cate_id,
                dropout_keep_prob: 1.0})
            pre = pre[:,1] # get probability of positive class
            predictions_dev.extend(pre.tolist())

        #acc = evaluation(predictions_dev, y_dev_label)
        #logging.info("Accuracy: {}".format(acc))

        auc = roc_auc_score(np.asarray(y_dev_label), np.asarray(predictions_dev))
        logging.info("Dev AUC: {}".format(auc))

        if auc > max_auc:
            save_path = saver.save(sess,"dead_in_month/results/model_1/model.weights/model.ckpt")
            logging.info("- new best score!")
            max_auc = auc
            current_early_stop_times = 0
        else:
            current_early_stop_times += 1
        if current_early_stop_times >= early_stop_times:
            logging.info("- early stopping {} epochs without improvement".format(current_early_stop_times))
            break

    predictions_test = []
    y_test_label = []
    for i in range(num_test_batch):
        tmp_test_file_name_list = test_file[i*n_batch:min((i+1)*n_batch, len(test_file))]
        tmp_y_test = y_test[i*n_batch:min((i+1)*n_batch, len(test_file))]
        y_test_label.extend(np.argmax(tmp_y_test,axis=1).tolist())
        tmp_x_test = []
        l=[]
        tmp_cate = []
        for f in tmp_test_file_name_list:
            new_x_test, new_l, new_cate = generate_token_embedding(f, mimic3_embedding)
            tmp_x_test.append(new_x_test)
            l.append(new_l)
            tmp_cate.append(new_cate)
        tmp_x_test = np.stack(tmp_x_test)
        cate_id = np.stack(tmp_cate)
        l = np.asarray(l)

        pre = sess.run(predictions,
                       feed_dict=
                       {input_x: tmp_x_test,
                        input_y: tmp_y_test,
                        sent_length: l,
                        category_index: cate_id,
                        dropout_keep_prob: 1.0})
        pre = pre[:,1]
        predictions_test.extend(pre.tolist())

    #acc = evaluation(predictions_test, y_test_label)
    #logging.info("Accuracy: {}".format(acc))

    auc = roc_auc_score(np.asarray(y_test_label), np.asarray(predictions_test))
    logging.info("AUC: {}".format(auc))


