from utility import generate_sent_category
import shutil
import glob
data_folder_path = '.../dead_in_month/file/category/'
pos_path = data_folder_path + 'pos/'
neg_path = data_folder_path + 'neg/'
pos_files = glob.glob(pos_path + "*.txt")
neg_files = glob.glob(neg_path + "*.txt")

print('truncate start')
save_file_path_pos = '.../dead_in_month/file/truncated_files_with_category/pos/'
save_file_path_neg = '.../dead_in_month/file/truncated_files_with_category/neg/'
for f in pos_files:
    sentences_per_file, _ = generate_sent_category(f)
    if len(sentences_per_file) <= 1000:
        #save file
        shutil.copy(f,save_file_path_pos)
print('finish pos')
for f in neg_files:
    sentences_per_file, _ = generate_sent_category(f)
    if len(sentences_per_file) <= 1000:

        shutil.copy(f,save_file_path_neg)

print('finish')
