# the functions used for extracting and preprocessing clinical notes from MIMIC-III
import pandas as pd
import re


def split_doc(d):
    """Split sentences in a document and saved the sentences to a list.
    
    Args:
        d: a document
        final_d: a list of sentences
    """
    
    d = d.strip().split(".") # split document by "." to sentences
    final_d = []
    for s in d:
        if s != "":  # ignore if the sentence is empty
            final_d.append(s.strip())
    return final_d  # Now the sentences are splitted from documents and saved to a list


def tokenize(sent, mimic3_embedding):
    """Tokenize the sentences accoring to the existing word from embedding. 
    
    Args:
        sent: input a sentence
        mimic3_embedding: find the existing word in embedding files
        cleaned_tokens: the tokens are cleaned and mapped to the mimic embedding 
    """
    
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
    """If the token is digit, then round the actual value into the nearest 10 times value.
    Args:
        s: original digit, 65 -> 60
        """
    if len(s) > 1:
        if s.isdigit():
            l = len(s)
            s = str(int(s)//(10**(l-1)) * 10**(l-1))
    return s.lower()


def get_age(row):
    """Calculate the age of patient by row
    Arg:
        row: the row of pandas dataframe. 
        return the patient age
    """
    raw_age = row['DOD'].year - row['DOB'].year
    if (row['DOD'].month < row['DOB'].month) or ((row['DOD'].month == row['DOB'].month) and (row['DOD'].day < row['DOB'].day)):
        return raw_age - 1
    else:
        return raw_age

def regenerate_dead_date(c):
    """Get the patient dead date
    Args: 
        c: row of pandas dataframe
        return the dead date if the patient is not dead in hospital, otherwise return dead date as -1.
    """
    if c['HOSPITAL_EXPIRE_FLAG'] == 1:
        return -1.0
    else:
        return c['dead_date']


def preprocess(df_note, admission, patient):
    """the utility to preprocess the mimic note
    Args:
        df_note: the note dataframe
        admission: admission csv file
        patient: patient csv file
        
        return: one dataframe with note as features and label of mortality or LOS
                patient subject id for later visualization
    """
    
    # remove discharge summary
    df_no_dis = df_note[df_note.CATEGORY != 'Discharge summary']
    # remove the note with error tag
    df_no_dis = df_no_dis[df_no_dis['ISERROR'] != 1]
    # drop the column that are not used in the future
    df_no_dis = df_no_dis.drop(['ROW_ID', 'STORETIME', 'DESCRIPTION', 'CGID', 'ISERROR'],
                               axis=1)

    # remove patient according to age
    patient['DOD'] = pd.to_datetime(patient['DOD'])
    patient['DOB'] = pd.to_datetime(patient['DOB'])
    patient['age'] = patient.apply(get_age, axis=1)
    patient = patient[(patient['age'].isnull()) | (patient['age'] >= 18)]
    # drop the column that are not used in the future
    patient = patient.drop(['ROW_ID', 'GENDER', 'DOD_SSN', 'EXPIRE_FLAG', 'age'], axis=1)

    # caculate the admission time for each patient
    admission['admit_times'] = admission.groupby(['SUBJECT_ID'])['SUBJECT_ID'].transform('size')
    # remove the patient with multiple admissions
    admission = admission[admission['admit_times'] < 2]
    # drop the column that are not used in the future
    admission = admission.drop(['ROW_ID', 'HADM_ID', 'ADMISSION_TYPE',
                                'ADMISSION_LOCATION', 'DISCHARGE_LOCATION',
                                'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS',
                                'ETHNICITY', 'EDREGTIME', 'EDOUTTIME',
                                'DIAGNOSIS', 'HAS_CHARTEVENTS_DATA', 'admit_times'], axis=1)

    # merge patient and admission csv to constraint patient
    patient_filter = pd.merge(patient, admission, on='SUBJECT_ID', how='inner')

    # merge patient and note; might generate a lot of replicate records
    patient_note = pd.merge(patient_filter, df_no_dis, on='SUBJECT_ID', how='inner')

    # remove chart after discharge
    patient_note['ADMITTIME'] = pd.to_datetime(patient_note['ADMITTIME'])
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

    patient_label = patient_note[['SUBJECT_ID', 'ADMITTIME','DOD', 'DISCHTIME',
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
    
    # calculate the dead date for patient
    patient_note_label['dead_date'] = patient_note_label['DOD'] - patient_note_label['DISCHTIME']
    # transfer time to date
    patient_note_label['dead_date'] = patient_note_label['dead_date'].dt.days

    patient_note_label['dead_after_disch_date'] = patient_note_label.apply(regenerate_dead_date,axis=1)
    
    # calculate the length of day for each patient
    patient_note_label['LOS'] = patient_note_label['DISCHTIME'] - patient_note_label['ADMITTIME']
    patient_note_label['LOS'] = patient_note_label['LOS'].dt.days
    
    # return the useful dataframe
    patient_note_with_label = patient_note_label[['full_text', 'dead_after_disch_date', "LOS"]]
    
    # also return the patient index
    patient_subjectid2index = patient_note_label['SUBJECT_ID']

    return patient_note_with_label, patient_subjectid2index
