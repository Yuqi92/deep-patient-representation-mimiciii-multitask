# extracting note and metadata from mimic csv

import pandas as pd
df_note = pd.read_csv('mimic_csv/NOTEEVENTS.csv')
admission = pd.read_csv('mimic_csv/ADMISSIONS.csv')
patient = pd.read_csv('mimic_csv/PATIENTS.csv')

# remove discharge summary according to the paper
df_no_dis = df_note[df_note.CATEGORY != 'Discharge summary']
df_no_dis = df_no_dis[df_no_dis['ISERROR'] != 1]
df_no_dis = df_no_dis.drop(['ROW_ID','STORETIME','DESCRIPTION','CGID','ISERROR'],axis=1)

# filter patient age >= 18
patient['DOD'] = pd.to_datetime(patient['DOD']) # change data type from string to datetime
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

# remove admission times more than once
admission['admit_times'] = admission.groupby(['SUBJECT_ID'])['SUBJECT_ID'].transform('size')
admission = admission[admission['admit_times'] < 2]
admission = admission.drop(['ROW_ID','HADM_ID','ADMITTIME','ADMISSION_TYPE','ADMISSION_LOCATION','DISCHARGE_LOCATION',
	'INSURANCE','LANGUAGE','RELIGION','MARITAL_STATUS','ETHNICITY','EDREGTIME','EDOUTTIME',	'DIAGNOSIS','HAS_CHARTEVENTS_DATA','admit_times'],axis=1)

# merge patient and admission csv table to firstly filter required patient
patient_filter = pd.merge(patient, admission, on='SUBJECT_ID', how='inner')

# merge patient and note; might generate a lot of replicate records
patient_note = pd.merge(patient_filter,df_no_dis, on='SUBJECT_ID', how='inner')

# remove chart recorded after discharge time according to the paper
patient_note['DISCHTIME'] = pd.to_datetime(patient_note['DISCHTIME'])
patient_note['CHARTDATE'] = pd.to_datetime(patient_note['CHARTDATE'])
patient_note['CHARTTIME'] = pd.to_datetime(patient_note['CHARTTIME'])
patient_note['DISCHDATE'] = patient_note['DISCHTIME'].values.astype('<M8[D]') # get date of discharge
# if charttime exists then use charttime< dischtime otherwise use chartdate<dischdate
patient_note = patient_note[(patient_note['CHARTTIME'] < patient_note['DISCHTIME']) | ((patient_note['CHARTDATE'] < patient_note['DISCHDATE']) & patient_note['CHARTTIME'].isnull())]
patient_note = patient_note.drop(['DOB','DOD_HOSP','HADM_ID','CHARTDATE','CHARTTIME'],axis=1)


# combine two columns: category and clinical note to one column in tuple type: (category, text)
patient_note['category_text'] = list(zip(patient_note['CATEGORY'], patient_note['TEXT']))

# generate patient(just one record)
patient_label = patient_note[['SUBJECT_ID','DOD','DISCHTIME','DEATHTIME','HOSPITAL_EXPIRE_FLAG']]
patient_label = patient_label.drop_duplicates()

# combine several note records of one patient into one entry: each patient will have just one record: ((cate1,text1),(cate2,text2)...)
note = patient_note[['SUBJECT_ID','category_text']]
aggregated = note.groupby('SUBJECT_ID')['category_text'].apply(tuple)
aggregated.name = 'full_text'
note = note.join(aggregated,on='SUBJECT_ID')

note = note.drop(['category_text'],axis=1)
note = note.drop_duplicates()


# generate two tables: patient with note, now the table has each patient with all of his/her notes&categories
patient_note_label = pd.merge(patient_label,note,on='SUBJECT_ID', how='inner')


# generate label for classification
## dead in hospital or not
patient_note_label['DEATHTIME'] = pd.to_datetime(patient_note_label['DEATHTIME'])
patient_note_label['DOD'] = pd.to_datetime(patient_note_label['DOD'])

def in_hosp(row):
    if ((row['DISCHTIME'] == row['DEATHTIME']) or (row['HOSPITAL_EXPIRE_FLAG'] == 1)):
        val = 1
    else:
        val = 0
    return val
patient_note_label['dead_in_hosp_label'] = patient_note_label.apply(in_hosp, axis=1)

## dead in one month after discharge
patient_note_label['dead_after_disch_date'] = patient_note_label['DOD'] - patient_note_label['DISCHTIME']
patient_note_label['dead_after_disch_date'] = patient_note_label['dead_after_disch_date'].dt.days

def month_after_hosp(row):
    if (row['dead_after_disch_date'] < 31) and (row['dead_after_disch_date'] > 0):
        val = 1
    else:
        val = 0
    return val
patient_note_label['month_after_hosp_label'] = patient_note_label.apply(month_after_hosp, axis=1)

## dead in one year after discharge
def year_after_hosp(row):
    if (row['dead_after_disch_date'] > 31) and (row['dead_after_disch_date'] < 366):
        val = 1
    else:
        val = 0
    return val
patient_note_label['year_after_hosp_label'] = patient_note_label.apply(year_after_hosp, axis=1)

# create each problem into one csv for later use

patient_note_in_hosp = patient_note_label[['SUBJECT_ID','full_text','dead_in_hosp_label']]

patient_note_in_month = patient_note_label[['SUBJECT_ID','full_text','month_after_hosp_label']]

patient_note_in_year = patient_note_label[['SUBJECT_ID','full_text','year_after_hosp_label']]

# extract text file for prediction model
for index,row in patient_note_in_hosp.iterrows():
    if row['dead_in_hosp_label'] == 0:
        f = open('file/category/neg/patient'+str(index)+'.txt','w')
        for x in row['full_text']:
            category = x[0]
            note_neg = x[1]
            f.write("****<<<<" + category +">>>>****"+'\n')
            f.write(note_neg)
        f.close()
    else:
        d = open('file/category/pos/patient'+str(index)+'.txt','w')
        for x in row['full_text']:
            category = x[0]
            note_pos = x[1]
            d.write("****<<<<" + category +">>>>****"+'\n')
            d.write(note_pos)
        d.close()




