import pandas as pd
import numpy as np
import HP
import glob

from preprocess_utilities import preprocess, split_doc, tokenize


def generate_label_3_task(c):
    """Generate label"""
    if c < 0:
        return 0
    elif (c >= 0 and c < 31):
        return 1
    elif (c >= 31 and c < 366):
        return 2
    else:
        return 3
    
def generate_label_5_task(c):
    """Generate label"""
    if c < 0:
        return 0
    elif (c >= 0 and c < 31):
        return 1
    elif (c >= 31 and c < 91):
        return 2
    elif (c >= 91 and c < 183):
        return 3
    elif (c >= 183 and c < 366):
        return 4
    else:
        return 5


def generate_label_20_task(c):
    """Generate label"""
    if c < 0:
        return 0
    elif (c >= 0 and c < 5):
        return 1
    elif (c >= 5 and c < 14):
        return 2
    elif (c >= 14 and c < 31):
        return 3
    elif (c >= 31 and c < 43):
        return 4
    elif (c >= 43 and c < 68):
        return 5
    elif (c >= 68 and c < 103):
        return 6
    elif (c >= 103 and c < 142):
        return 7
    elif (c >= 142 and c < 196):
        return 8
    elif (c >= 196 and c < 269):
        return 9
    elif (c >= 269 and c < 366):
        return 10
    elif (c >= 366 and c < 453):
        return 11
    elif (c >= 453 and c < 573):
        return 12
    elif (c >= 573 and c < 711):
        return 13
    elif (c >= 711 and c < 893):
        return 14
    elif (c >= 893 and c < 1092):
        return 15
    elif (c >= 1092 and c < 1342):
        return 16
    elif (c >= 1342 and c < 1626):
        return 17
    elif (c >= 1626 and c < 1997):
        return 18
    elif (c >= 1997 and c < 2548):
        return 19
    else:
        return 20
    
    
# generate patient label for visualization
df_note = pd.read_csv(HP.mimic_note_events)
admission = pd.read_csv(HP.mimic_admissions)
patient = pd.read_csv(HP.mimic_patients)

patient_note_label, patient_subjectid2index = preprocess(df_note, admission, patient)

test_index = np.load(HP.index_test_path)

test_note = patient_note_label.loc[test_index]

test_note = test_note.reset_index(drop = False)

test_note['label'] = test_note['dead_after_disch_date'].apply(generate_label_3_task)

test_patient_label = test_note[['index','label']]

# generate test patient vector
path_to_data = HP.patient_vector_directory

npy_file_list = glob.iglob(path_to_data+'*.npy')

all_patient = {}

for npy_file in npy_file_list:
    patient_id = npy_file[91:-4] # need to be changed when 20-task
    patient_vec = np.load(npy_file)
    all_patient[patient_id] = patient_vec

test_patient = {}
for k,v in all_patient.items():
    k = int(k)
    if k in test_index:
        test_patient[k] = v

test_patient_csv = pd.DataFrame.from_dict(test_patient, orient='index')
test_patient_csv = test_patient_csv.reset_index()

# associate patient vector and label together

vector_label = pd.merge(test_patient_csv,test_patient_label,on='index')


list_vector = []
for index, row in vector_label.iterrows():
    list_vector.append(np.asarray(row[1:51]))

array_vector = np.stack(list_vector)

np.save(HP.tsne_vector_directory, array_vector)


f = open(HP.tsne_label_directory,'w')
for id, row in vector_label.iterrows():
    f.write(str(row['label']) + '\n')
f.close()
