import os
import glob
import tensorflow as tf
import numpy as np
import pandas as pd
import math
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from utility import  extract_embedding, generate_token_embedding, split_train_test_dev, CNN_model,generate_label_from_dead_date

import logging
logging.basicConfig(filename="example.log", level=logging.INFO)


logging.info("processing mimic")
df_note = pd.read_csv('mimic_csv/NOTEEVENTS.csv')
admission = pd.read_csv('mimic_csv/ADMISSIONS.csv')
patient = pd.read_csv('mimic_csv/PATIENTS.csv')

# remove discharge summary
df_no_dis = df_note[df_note.CATEGORY != 'Discharge summary']
df_no_dis = df_no_dis[df_no_dis['ISERROR'] != 1]
df_no_dis = df_no_dis.drop(['ROW_ID','STORETIME','DESCRIPTION','CGID','ISERROR'],axis=1)
# remove patient according to age
patient['DOD'] = pd.to_datetime(patient['DOD'])
patient['DOB'] = pd.to_datetime(patient['DOB'])

def get_age(row):
    raw_age = row['DOD'].year - row['DOB'].year
    if (row['DOD'].month < row['DOB'].month) or ((row['DOD'].month == row['DOB'].month) and (row['DOD'].day < row['DOB'].day)):
        return raw_age - 1
    else:
        return raw_age
patient['age'] = patient.apply(get_age, axis=1)
patient = patient[(patient['age'].isnull())|(patient['age'] >= 18)]

patient = patient.drop(['ROW_ID','GENDER','DOD_SSN','EXPIRE_FLAG','age'],axis=1)

#admit time = 1
admission['admit_times'] = admission.groupby(['SUBJECT_ID'])['SUBJECT_ID'].transform('size')
admission = admission[admission['admit_times'] < 2]
admission = admission.drop(['ROW_ID','HADM_ID','ADMITTIME','ADMISSION_TYPE','ADMISSION_LOCATION','DISCHARGE_LOCATION',
	'INSURANCE','LANGUAGE','RELIGION','MARITAL_STATUS','ETHNICITY','EDREGTIME','EDOUTTIME',
	'DIAGNOSIS','HAS_CHARTEVENTS_DATA','admit_times'],axis=1)

#merge patient and admission csv to constraint patient
patient_filter = pd.merge(patient, admission, on='SUBJECT_ID', how='inner')

#merge patient and note; might generate a lot of replicate records
patient_note = pd.merge(patient_filter,df_no_dis, on='SUBJECT_ID', how='inner')

#remove chart after discharge
patient_note['DISCHTIME'] = pd.to_datetime(patient_note['DISCHTIME'])
patient_note['CHARTDATE'] = pd.to_datetime(patient_note['CHARTDATE'])
patient_note['CHARTTIME'] = pd.to_datetime(patient_note['CHARTTIME'])
patient_note['DISCHDATE'] = patient_note['DISCHTIME'].values.astype('<M8[D]')
patient_note = patient_note[(patient_note['CHARTTIME'] < patient_note['DISCHTIME']) | ((patient_note['CHARTDATE'] < patient_note['DISCHDATE']) & patient_note['CHARTTIME'].isnull())]
patient_note = patient_note.drop(['DOB','DOD_HOSP','HADM_ID','CHARTDATE','CHARTTIME'],axis=1)

# combine two columns to one column with tuple
patient_note['category_text'] = list(zip(patient_note['CATEGORY'], patient_note['TEXT']))

patient_label = patient_note[['SUBJECT_ID','DOD','DISCHTIME','DEATHTIME','HOSPITAL_EXPIRE_FLAG']]
patient_label = patient_label.drop_duplicates()

# combine several duplicate records along the column into one entry
note = patient_note[['SUBJECT_ID','category_text']]
aggregated = note.groupby('SUBJECT_ID')['category_text'].apply(tuple)
aggregated.name = 'full_text'
note = note.join(aggregated,on='SUBJECT_ID')

note = note.drop(['category_text'],axis=1)
note = note.drop_duplicates()

patient_note_label = pd.merge(patient_label,note,on='SUBJECT_ID', how='inner')


#patient_note_label['DISCHTIME'] = pd.to_datetime(patient_note_label['DISCHTIME'])
patient_note_label['DEATHTIME'] = pd.to_datetime(patient_note_label['DEATHTIME'])
patient_note_label['DOD'] = pd.to_datetime(patient_note_label['DOD'])


patient_note_label['dead_after_disch_date'] = patient_note_label['DOD'] - patient_note_label['DISCHTIME']
patient_note_label['dead_after_disch_date'] = patient_note_label['dead_after_disch_date'].dt.days

clinical_note = patient_note_label[['full_text']]
dead_after_disch_date = patient_note_label[['dead_after_disch_date']]

# get index
logging.info('get index')
patient_index_list = dead_after_disch_date.index.tolist()

# split train test dev
logging.info('split_train_test_dev')
load_path = '/home/ysi/PycharmProjects/CNN_mimic_iii/multi_task/index'
train_index,test_index,dev_index = split_train_test_dev(patient_index_list,load_path,load=False)

dev_dead_date = dead_after_disch_date['dead_after_disch_date'].iloc[dev_index]
test_dead_date = dead_after_disch_date['dead_after_disch_date'].iloc[test_index]
train_dead_date = dead_after_disch_date['dead_after_disch_date'].iloc[train_index]

logging.info('generate label from dead date')
task1 = 0
y_dev_task1 = generate_label_from_dead_date(dev_dead_date,task1)
y_test_task1 = generate_label_from_dead_date(test_dead_date,task1)
y_train_task1 = generate_label_from_dead_date(train_dead_date,task1)
task2 = 31
y_dev_task2 = generate_label_from_dead_date(dev_dead_date,task2)
y_test_task2 = generate_label_from_dead_date(test_dead_date,task2)
y_train_task2 = generate_label_from_dead_date(train_dead_date,task2)
task3 = 366
y_dev_task3 = generate_label_from_dead_date(dev_dead_date,task3)
y_test_task3 = generate_label_from_dead_date(test_dead_date,task3)
y_train_task3 = generate_label_from_dead_date(train_dead_date,task3)

logging.info('get files')
files_folder = '/home/ysi/PycharmProjects/CNN_mimic_iii/file/'
#files = glob.glob(files_folder + "*.txt")
files = []
for i in range(len(patient_index_list)):
    files.append(files_folder + "patient" + str(i) + ".txt")

document_name_list = np.asarray(files)

dev_file = document_name_list[dev_index]
test_file = document_name_list[test_index]
train_file = document_name_list[train_index]

# generate mimic embedding
logging.info('extract mimic')
embedding_folder = '/home/ysi/Documents/amia/cancer_ner_relation_v1/data/glove.6B'
file_embedding= open(os.path.join(embedding_folder,'mimic.k100.w2v'))
mimic3_embedding = extract_embedding(file_embedding)

# train CNN model
multi_size = 3
embedding_size = 100
max_document_length = 1000
max_sentence_length = 25
n_class = 2
n_batch = 64
early_stop_times = 5

num_train_batch = int(math.ceil(len(train_file) / n_batch))
num_dev_batch = int(math.ceil(len(dev_file) / n_batch))
num_test_batch = int(math.ceil(len(test_file) / n_batch))

input_x = tf.placeholder(tf.float32, [None, max_document_length, max_sentence_length,embedding_size], name="input_x")
input_ys = []
for i in range(multi_size):
    input_ys.append(tf.placeholder(tf.int32, [None, n_class], name="input_y"+str(i)))

sent_length = tf.placeholder(tf.int32, [None], name="sent_length")

# category placeholder
category_index = tf.placeholder(tf.int32, [None, max_document_length], name='category_index')
dropout_keep_prob = tf.placeholder(tf.float32, [],name="dropout_keep_prob")
#lr_placeholder = tf.placeholder(tf.float32, [],name="lr")

optimize, scores_soft_max_list = CNN_model(input_x, input_ys, sent_length, category_index, dropout_keep_prob)
saver = tf.train.Saver()

with tf.Session() as sess:
    restore = False
    if restore:
        saver.restore(sess, "multi_task/results/model_1/model.weights/model.ckpt")
    else:
        sess.run(tf.global_variables_initializer())
    shuf_ind = np.asarray(list(range(len(train_file))))
    max_auc = 0
    current_early_stop_times = 0

    while True:
        np.random.shuffle(shuf_ind)
        train_file = train_file[shuf_ind]

        y_train_task1_shuf = y_train_task1[shuf_ind]
        y_train_task2_shuf = y_train_task2[shuf_ind]
        y_train_task3_shuf = y_train_task3[shuf_ind]

        # start train
        for i in tqdm(range(num_train_batch)):
            tmp_train_file_name_list = train_file[i*n_batch:min((i+1)*n_batch, len(train_file))]
            tmp_y_train_task1 = y_train_task1_shuf[i*n_batch:min((i+1)*n_batch, len(train_file))]
            tmp_y_train_task2 = y_train_task2_shuf[i*n_batch:min((i+1)*n_batch, len(train_file))]
            tmp_y_train_task3 = y_train_task3_shuf[i*n_batch:min((i+1)*n_batch, len(train_file))]

            tmp_y_train = [tmp_y_train_task1,tmp_y_train_task2,tmp_y_train_task3]

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
            feed_dict = {input_x: tmp_x_train,
                         sent_length: l,
                         category_index: cate_id,
                         dropout_keep_prob: 0.8}
            for (M,input_y) in enumerate(input_ys):
                feed_dict[input_y] = tmp_y_train[M]
            sess.run([optimize],feed_dict=feed_dict)


        # get validation result
        y_dev_label = []

        predictions_dev = []

        for i in range(num_dev_batch):
            tmp_dev_file_name_list = dev_file[i*n_batch:min((i+1)*n_batch, len(dev_file))]
            tmp_y_dev_task1 = y_dev_task1[i*n_batch:min((i+1)*n_batch, len(dev_file))]
            tmp_y_dev_task2 = y_dev_task2[i*n_batch:min((i+1)*n_batch, len(dev_file))]
            tmp_y_dev_task3 = y_dev_task3[i*n_batch:min((i+1)*n_batch, len(dev_file))]
            tmp_y_dev = [tmp_y_dev_task1,tmp_y_dev_task2,tmp_y_dev_task3]

            y_dev_label.extend(np.argmax(tmp_y_dev_task1,axis=1).tolist())
            y_dev_label.extend(np.argmax(tmp_y_dev_task2,axis=1).tolist())
            y_dev_label.extend(np.argmax(tmp_y_dev_task3,axis=1).tolist()) #Y_DEV_LABEL SHAPE: [3*n_batch,1]
            # order of y_dev_label: task1,2,3_batch1 + task1,2,3_batch2 +...
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
            feed_dict = {input_x: tmp_x_dev,
                         sent_length: l,
                         category_index: cate_id,
                         dropout_keep_prob: 1.0}
            for (M,input_y) in enumerate(input_ys):
                feed_dict[input_y] = tmp_y_dev[M]
            pre = sess.run(scores_soft_max_list,feed_dict=feed_dict) #[3,n_batch,2]
            pre = np.asarray(pre)
            pre = pre.reshape(-1,n_class)                         #[3*n_batch,2]  in one batch: task1+task2+task3
            pre = pre[:,1] # get probability of positive class
            predictions_dev.extend(pre.tolist())   #task1,2,3_batch1 + task1,2,3_batch2+ task1,2,3_batch3....

        #acc = evaluation(predictions_dev, y_dev_label)
        #logging.info("Accuracy: {}".format(acc))

        auc = roc_auc_score(np.asarray(y_dev_label), np.asarray(predictions_dev))
        logging.info("Dev AUC: {}".format(auc))

        if auc > max_auc:
            save_path = saver.save(sess,"multi_task/results/model_1/model.weights/model.ckpt")
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
        tmp_y_test_task1 = y_test_task1[i*n_batch:min((i+1)*n_batch, len(test_file))]
        tmp_y_test_task2 = y_test_task2[i*n_batch:min((i+1)*n_batch, len(test_file))]
        tmp_y_test_task3 = y_test_task3[i*n_batch:min((i+1)*n_batch, len(test_file))]
        tmp_y_test = [tmp_y_test_task1,tmp_y_test_task2,tmp_y_test_task3]

        y_test_label.extend(np.argmax(tmp_y_test_task1,axis=1).tolist())
        y_test_label.extend(np.argmax(tmp_y_test_task2,axis=1).tolist())
        y_test_label.extend(np.argmax(tmp_y_test_task3,axis=1).tolist())

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
        feed_dict = {input_x: tmp_x_test,
                     sent_length: l,
                     category_index: cate_id,
                     dropout_keep_prob: 1.0}
        for (M,input_y) in enumerate(input_ys):
                feed_dict[input_y] = tmp_y_test[M]

        pre = sess.run(scores_soft_max_list,feed_dict=feed_dict)
        pre = np.asarray(pre)
        pre = pre.reshape(-1,n_class)
        pre = pre[:,1]
        predictions_test.extend(pre.tolist())

    #acc = evaluation(predictions_test, y_test_label)
    #logging.info("Accuracy: {}".format(acc))

    auc = roc_auc_score(np.asarray(y_test_label), np.asarray(predictions_test))
    logging.info("AUC: {}".format(auc))




