import os
import HP

def main():
    # create directory for patient
    if not os.path.exists(HP.insight_patient_data_directory):
        os.makedirs(HP.insight_patient_data_directory)
    else:
        raise IOError
    # get the patient file and store it into multiple files (one file per sentence)
    f = open(HP.data_directory + HP.insight_patient_id + ".txt")
    current_sentence_id = 0
    f_to_write = open(HP.insight_patient_data_directory + "sentence" + str(current_sentence_id) + ".txt", "w")
    for line in f:
        if len(line.strip()) != 0:
            f_to_write.write(line)
        else:
            f_to_write.write(line)
            f_to_write.close()
            current_sentence_id += 1
            f_to_write = open(HP.insight_patient_data_directory + "sentence" + str(current_sentence_id) + ".txt", "w")
    f_to_write.close()
    f.close()
    total_sentence_n = current_sentence_id
    # get result_csv for specific patient
    insight_patient_result_csv_file = open(HP.insight_patient_result_csv, "w")
    result_csv_file = open(HP.result_csv)
    header = result_csv_file.readline()
    insight_patient_result_csv_file.write(header)
    for line in result_csv_file:
        if line.startswith(HP.insight_patient_id + ","):
            line_after_patient_id = line[len(HP.insight_patient_id):]
            for i in range(total_sentence_n):
                insight_patient_result_csv_file.write("sentence" + str(i) + line_after_patient_id)
            break

if __name__ == "__main__":
    main()
