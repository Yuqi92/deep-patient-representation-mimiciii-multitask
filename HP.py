
# data
mimic_note_events = '/home/ysi/PycharmProjects/CNN_mimiciii_mortality_prediction/mimic_csv/NOTEEVENTS.csv'
mimic_admissions = '/home/ysi/PycharmProjects/CNN_mimiciii_mortality_prediction/mimic_csv/ADMISSIONS.csv'
mimic_patients = '/home/ysi/PycharmProjects/CNN_mimiciii_mortality_prediction/mimic_csv/PATIENTS.csv'

result_csv = "/home/ysi/PycharmProjects/CNN_mimiciii_mortality_prediction/merged/file/result.csv"
data_directory = "/home/ysi/PycharmProjects/CNN_mimiciii_mortality_prediction/merged/file/entire_file/"
patient_vector_directory = "/home/ysi/PycharmProjects/CNN_mimiciii_mortality_prediction/merged/patient_vector/"

#result_csv = "/home/ysi/PycharmProjects/CNN_mimiciii_mortality_prediction/out_hosp/result.csv"
#data_directory = "/home/ysi/PycharmProjects/CNN_mimiciii_mortality_prediction/out_hosp/file/"

index_path = '/home/ysi/PycharmProjects/CNN_mimiciii_mortality_prediction/merged/index'
#index_path = '/home/ysi/PycharmProjects/CNN_mimiciii_mortality_prediction/out_hosp/index'
index_train_path = index_path + '/train.npy'
index_dev_path = index_path + '/dev.npy'
index_test_path = index_path + '/test.npy'
load_index = True
read_data_thread_num = 8

embedding_file = '/home/ysi/Documents/amia/cancer_ner_relation_v1/data/glove.6B/mimic.k100.w2v'

n_max_sentence_num = 1000 # truncated to 1000 sentences a document
n_max_word_num = 25 # truncated to 25 words a sentence

# category
category = ['pad', 'Respiratory', 'ECG','Radiology','Nursing/other','Rehab Services','Nutrition','Pharmacy','Social Work',
            'Case Management','Physician','General','Nursing','Echo','Consult']
category_id = {cate: idx for idx, cate in enumerate(category)}


# task
tasks_dead_date = [0, 31, 366]
# 20-task
# tasks_dead_date = [0,5,14,31,43,68,103,142,196,269,366,453,573,711,893,1092,1342,1626,1997,2548]
# 
# tasks_dead_date = [0, 31, 91, 183, 366]


# model
restore = False
multi_size = len(tasks_dead_date)
embedding_size = 100
max_document_length = 1000
max_sentence_length = 25
n_batch = 64
early_stop_times = 5
filter_sizes = (3, 4, 5)
num_filters = 50
n_category = len(category)
dim_category = 10
document_filter_size = 3
document_num_filters = 50
learning_rate = 0.01

num_classes = 2
lambda_regularizer_strength = 5

model_path = "merged/multi_task/result_3_task/model_1/model.weights/model.ckpt"
#out_hosp/results/uni_task_in_month/model_1/model.weights/model.ckpt"
# CNN_mimiciii_mortality_prediction/merged/multi_task/result_20_task/model_1
# log
log_file_name = "example.log"


