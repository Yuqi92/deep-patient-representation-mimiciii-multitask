import tensorflow as tf
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from utility import load_x_data_for_cnn, split_train_test_dev, CNN_model,\
    generate_label_from_dead_date, generate_label_from_los_date, generate_label_from_date,\
    test_dev_auc, simple_model, load_x_data_for_simple
import logging
import HP
from Embedding import Embedding

_ = Embedding.get_embedding()

# log for the project
logging.basicConfig(filename=HP.log_file_name, level=logging.INFO, format='%(asctime)s %(message)s')

result_csv = pd.read_csv(HP.get_result_csv())

# split train test dev
logging.info('split_train_test_dev')
train_index, test_index, dev_index = split_train_test_dev(len(result_csv))

# get the label for train test and dev
train_dead_date = result_csv['dead_after_disch_date'].iloc[train_index]
dev_dead_date = result_csv['dead_after_disch_date'].iloc[dev_index]
test_dead_date = result_csv['dead_after_disch_date'].iloc[test_index]

train_los_date = result_csv['length_of_stay'].iloc[train_index]
dev_los_date = result_csv['length_of_stay'].iloc[dev_index]
test_los_date = result_csv['length_of_stay'].iloc[test_index]

dev_patient_name = np.asarray(result_csv["patient_id"].iloc[dev_index])
test_patient_name = np.asarray(result_csv["patient_id"].iloc[test_index])  # patient0, patient1, ...
train_patient_name = np.asarray(result_csv["patient_id"].iloc[train_index])

logging.info('generate label from dead date')

if len(HP.tasks_dead_date) != 0 and len(HP.tasks_los_date) != 0:
    y_dev_task = generate_label_from_date(dev_dead_date, dev_los_date)  # list of nparray
    y_test_task = generate_label_from_date(test_dead_date, test_los_date)
    y_train_task = generate_label_from_date(train_dead_date, train_los_date)
elif len(HP.tasks_los_date) == 0:
    y_dev_task = generate_label_from_dead_date(dev_dead_date)  # list of nparray
    y_test_task = generate_label_from_dead_date(test_dead_date)
    y_train_task = generate_label_from_dead_date(train_dead_date)
elif len(HP.tasks_dead_date) == 0:
    y_dev_task = generate_label_from_los_date(dev_los_date)  # list of nparray
    y_test_task = generate_label_from_los_date(test_los_date)
    y_train_task = generate_label_from_los_date(train_los_date)
else:
    logging.error("no task label")
    y_dev_task = None
    y_test_task = None
    y_train_task = None


n_train = len(train_patient_name)
n_dev = len(dev_patient_name)
n_test = len(test_patient_name)

# train CNN model
num_train_batch = int(math.ceil(n_train / HP.n_batch))
num_dev_batch = int(math.ceil(n_dev / HP.n_batch))
num_test_batch = int(math.ceil(n_test / HP.n_batch))

# define placeholders and model
input_ys = []
for i in range(HP.multi_size):
    input_ys.append(tf.placeholder(tf.int32, [None, HP.num_classes], name="input_y"+str(i)))

if HP.model_type == "CNN":
    input_x = tf.placeholder(tf.float32,
                         [None, HP.max_document_length, HP.max_sentence_length, HP.embedding_size],
                         name="input_x")
    sent_length = tf.placeholder(tf.int32, [None], name="sent_length")
    # category placeholder
    category_index = tf.placeholder(tf.int32, [None, HP.max_document_length], name='category_index')
    dropout_keep_prob = tf.placeholder(tf.float32, [], name="dropout_keep_prob")
    optimize, scores_soft_max_list, _ = CNN_model(input_x, input_ys, sent_length, category_index, dropout_keep_prob)

elif HP.model_type == "SIMPLE":
    input_x = tf.placeholder(tf.float32,
                         [None, HP.document_num_filters],
                         name="input_x")
    sent_length = None
    category_index = None
    dropout_keep_prob = None
    optimize, scores_soft_max_list = simple_model(input_x, input_ys)
else:
    logging.error("unsupport model type")
    optimize = None
    scores_soft_max_list = None
saver = tf.train.Saver()

# start tf
with tf.Session() as sess:
    if HP.restore:
        saver.restore(sess, HP.model_path)
    else:
        sess.run(tf.global_variables_initializer())
    
    shuf_train_ind = np.arange(n_train)
    max_auc = 0
    current_early_stop_times = 0

    while not HP.use_everything_to_test:
        np.random.shuffle(shuf_train_ind)
        train_patient_name = train_patient_name[shuf_train_ind]
        for i in range(len(y_train_task)):
            y_train_task[i] = y_train_task[i][shuf_train_ind]

        # start train
        for i in tqdm(range(num_train_batch)):
            # logging.info("start new batch")
            tmp_train_patient_name = train_patient_name[i*HP.n_batch:min((i+1)*HP.n_batch, n_train)]
            tmp_y_train = []
            for t in y_train_task:
                tmp_y_train.append(t[i*HP.n_batch:min((i+1)*HP.n_batch, n_train)])

            if HP.model_type == "CNN":
                feed_dict = load_x_data_for_cnn(tmp_train_patient_name,
                                                HP.drop_out_train,
                                                input_x,
                                                sent_length,
                                                category_index,
                                                dropout_keep_prob)
            elif HP.model_type == "SIMPLE":
                feed_dict = load_x_data_for_simple(tmp_train_patient_name, input_x)
            else:
                logging.error("unsupported model type")
                feed_dict = None

            for (M, input_y) in enumerate(input_ys):
                feed_dict[input_y] = tmp_y_train[M]
            # logging.info("start to train")
            sess.run([optimize], feed_dict=feed_dict)

        # get validation result
        dev_auc,_ = test_dev_auc(num_dev_batch, y_dev_task, dev_patient_name, n_dev, sess,
                               input_x, sent_length, category_index, dropout_keep_prob, scores_soft_max_list, test_output_flag=False)
        logging.info("Dev AUC: {}".format(dev_auc))
        
        # early stop techniques
        if dev_auc > max_auc:
            save_path = saver.save(sess, HP.model_path)
            logging.info("- new best score!")
            max_auc = dev_auc
            current_early_stop_times = 0
        else:
            current_early_stop_times += 1
        if current_early_stop_times >= HP.early_stop_times:
            logging.info("- early stopping {} epochs without improvement".format(current_early_stop_times))
            break
    
    # report the performance on test data
    test_auc,test_auc_per_task = test_dev_auc(num_test_batch, y_test_task, test_patient_name, n_test, sess,
                            input_x, sent_length, category_index, dropout_keep_prob, scores_soft_max_list, test_output_flag=True)
    logging.info("Test total AUC: {}".format(test_auc))
    logging.info("Test total AUC: {}".format(test_auc_per_task))


