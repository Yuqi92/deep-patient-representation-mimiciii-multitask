import pandas as pd
import re


def split_doc(d):
    d = d.strip().split(".") # split document by "." to sentences
    final_d = []
    for s in d:
        if s != "":  # ignore if the sentence is empty
            final_d.append(s.strip())
    return final_d  # Now the sentences are splitted from documents and saved to a list


def tokenize(sent, mimic3_embedding):
    tokenizer = re.compile('\w+|\*\*|[^\s\w]')
    tokens = tokenizer.findall(sent.lower())
    cleaned_tokens = []
    for tok in tokens:
        tok = _clean_token(tok)
        if tok in mimic3_embedding:
            cleaned_tokens.append(tok)
        else:
            cleaned_tokens.append('UNK')
    return cleaned_tokens


def _clean_token(s):
    if len(s) > 1:
        if s.isdigit():
            l = len(s)
            s = str(int(s)//(10**(l-1)) * 10**(l-1))
    return s.lower()


def get_age(row):
    raw_age = row['DOD'].year - row['DOB'].year
    if (row['DOD'].month < row['DOB'].month) or ((row['DOD'].month == row['DOB'].month) and (row['DOD'].day < row['DOB'].day)):
        return raw_age - 1
    else:
        return raw_age


def preprocess(df_note, admission, patient):
    # remove discharge summary
    df_no_dis = df_note[df_note.CATEGORY != 'Discharge summary']
    df_no_dis = df_no_dis[df_no_dis['ISERROR'] != 1]
    df_no_dis = df_no_dis.drop(['ROW_ID', 'STORETIME', 'DESCRIPTION', 'CGID', 'ISERROR'],
                               axis=1)

    # remove patient according to age
    patient['DOD'] = pd.to_datetime(patient['DOD'])
    patient['DOB'] = pd.to_datetime(patient['DOB'])
    patient['age'] = patient.apply(get_age, axis=1)
    patient = patient[(patient['age'].isnull()) | (patient['age'] >= 18)]
    patient = patient.drop(['ROW_ID', 'GENDER', 'DOD_SSN', 'EXPIRE_FLAG', 'age'], axis=1)

    # admit time = 1
    admission['admit_times'] = admission.groupby(['SUBJECT_ID'])['SUBJECT_ID'].transform('size')
    admission = admission[admission['admit_times'] < 2]
    admission = admission.drop(['ROW_ID', 'HADM_ID', 'ADMITTIME', 'ADMISSION_TYPE',
                                'ADMISSION_LOCATION', 'DISCHARGE_LOCATION',
                                'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS',
                                'ETHNICITY', 'EDREGTIME', 'EDOUTTIME',
                                'DIAGNOSIS', 'HAS_CHARTEVENTS_DATA', 'admit_times'], axis=1)

    #merge patient and admission csv to constraint patient
    patient_filter = pd.merge(patient, admission, on='SUBJECT_ID', how='inner')

    #merge patient and note; might generate a lot of replicate records
    patient_note = pd.merge(patient_filter, df_no_dis, on='SUBJECT_ID', how='inner')

    #remove chart after discharge
    patient_note['DISCHTIME'] = pd.to_datetime(patient_note['DISCHTIME'])
    patient_note['CHARTDATE'] = pd.to_datetime(patient_note['CHARTDATE'])
    patient_note['CHARTTIME'] = pd.to_datetime(patient_note['CHARTTIME'])
    patient_note['DISCHDATE'] = patient_note['DISCHTIME'].values.astype('<M8[D]')
    patient_note = patient_note[(patient_note['CHARTTIME'] < patient_note['DISCHTIME']) |
                                ((patient_note['CHARTDATE'] < patient_note['DISCHDATE']) &
                                 patient_note['CHARTTIME'].isnull())]
    patient_note = patient_note.drop(['DOB', 'DOD_HOSP', 'HADM_ID', 'CHARTDATE', 'CHARTTIME'],
                                     axis=1)

    # combine two columns to one column with tuple
    patient_note['category_text'] = list(zip(patient_note['CATEGORY'], patient_note['TEXT']))

    patient_label = patient_note[['SUBJECT_ID', 'DOD', 'DISCHTIME',
                                  'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG']]
    patient_label = patient_label.drop_duplicates()

    # combine several duplicate records along the column into one entry
    note = patient_note[['SUBJECT_ID','category_text']]
    aggregated = note.groupby('SUBJECT_ID')['category_text'].apply(tuple)
    aggregated.name = 'full_text'
    note = note.join(aggregated, on='SUBJECT_ID')

    note = note.drop(['category_text'], axis=1)
    note = note.drop_duplicates()

    patient_note_label = pd.merge(patient_label, note, on='SUBJECT_ID', how='inner')

    patient_note_label['DEATHTIME'] = pd.to_datetime(patient_note_label['DEATHTIME'])
    patient_note_label['DOD'] = pd.to_datetime(patient_note_label['DOD'])

    patient_note_label['dead_after_disch_date'] = patient_note_label['DOD'] - patient_note_label['DISCHTIME']
    patient_note_label['dead_after_disch_date'] = patient_note_label['dead_after_disch_date'].dt.days

    patient_note_label = patient_note_label[['full_text', 'dead_after_disch_date']]

    return patient_note_label
