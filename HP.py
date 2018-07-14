
# data
mimic_note_events = 'mimic_csv/NOTEEVENTS.csv'
mimic_admissions = 'mimic_csv/ADMISSIONS.csv'
mimic_patients = 'mimic_csv/PATIENTS.csv'

result_csv = "file/result.csv"
data_directory = "file/"
index_path = 'multi_task/index'
index_train_path = index_path + '/train.npy'
index_dev_path = index_path + '/dev.npy'
index_test_path = index_path + '/test.npy'
load_index = False


embedding_file = '/home/ysi/Documents/amia/cancer_ner_relation_v1/data/glove.6B/mimic.k100.w2v'

n_max_sentence_num = 1000 # truncated to 1000 sentences a document
n_max_word_num = 25 # truncated to 25 words a sentence

# category
category = ['pad', 'Respiratory', 'ECG','Radiology','Nursing/other','Rehab Services','Nutrition','Pharmacy','Social Work',
            'Case Management','Physician','General','Nursing','Echo','Consult']
category_id = {cate: idx for idx, cate in enumerate(category)}


# task
tasks_dead_date = [0, 31, 366]

# model
restore = False
multi_size = len(tasks_dead_date)
embedding_size = 100
max_document_length = 1000
max_sentence_length = 25
n_class = 2
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

model_path = "multi_task/results/model_1/model.weights/model.ckpt"

# log
log_file_name = "example.log"


