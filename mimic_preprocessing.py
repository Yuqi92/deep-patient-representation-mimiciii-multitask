# extracting note and metadata from mimic csv
import pandas as pd
import HP
from preprocess_utilities import preprocess, split_doc, tokenize
from Embedding import Embedding

df_note = pd.read_csv(HP.mimic_note_events)
admission = pd.read_csv(HP.mimic_admissions)
patient = pd.read_csv(HP.mimic_patients)

patient_note_label = preprocess(df_note, admission, patient)

mimic3_embedding = Embedding.get_embedding()

# extract text file for prediction model
result = open(HP.result_csv, "w")
result.write("patient_id,dead_after_disch_date\n")
for index, row in patient_note_label.iterrows():
    tmp_full_text = row['full_text']
    tmp_dead_after_disch_date = row["dead_after_disch_date"]
    tmp_patient_id = "patient" + str(index)
    result.write(tmp_patient_id + "," + str(tmp_dead_after_disch_date) + "\n")
    f = open(HP.data_directory + tmp_patient_id + '.txt', 'w')
    for x in row['full_text']:
        category = x[0].strip()
        category_index = HP.category_id[category]
        doc = x[1]
        sentences = split_doc(doc)
        for sent in sentences:
            cleaned_tokens = tokenize(sent, mimic3_embedding)
            if len(cleaned_tokens) > 0:
                f.write(str(category_index) + "\n")
                for t in cleaned_tokens:
                    f.write(t + "\n")
                f.write("\n")
    f.close()
result.close()
