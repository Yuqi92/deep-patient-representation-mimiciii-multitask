import re
import tensorflow as tf
import numpy as np

# generate category id
category = ['pad','Respiratory','ECG','Radiology','Nursing/other','Rehab Services','Nutrition','Pharmacy','Social Work',
            'Case Management','Physician','General','Nursing','Echo','Consult']

category_id = {cate: idx for idx, cate in enumerate(category)}

# generate pre-trained embedding
def extract_embedding(embedding_file):
    embedding_map = {}
    for line in embedding_file:
        values = line.split()
        word = values[0]
        coef = np.asarray(values[1:], dtype = 'float32')
        embedding_map[word] = coef
    embedding_file.close()
    return embedding_map

# extract and save sentence and category from each text file

def generate_sent_category(file_name):
    n_max_sentence_num = 1000
    def split_doc(d):
        d = d.strip().split(".") # split document by "." to sentences
        final_d = []
        for s in d:
            if s != "":  # ignore if the sentence is empty
                final_d.append(s.strip())
        return final_d  # Now the sentences are splitted from documents and saved to a list

    with open(file_name,'r') as f:
        sentences = []
        categories = []
        current_category = ""
        new_doc = ""
        for line in f:
            line = line.strip()
            if line.startswith("****<<<<"):
                if new_doc != "":
                    tmp_split_doc = split_doc(new_doc) # all the sentences under this category are saved to a list
                    tmp_sent_count = len(tmp_split_doc) # calculate how many sentences mean that how many categories have to generate
                    sentences += tmp_split_doc  # save the sentence list
                    categories += ([current_category] * tmp_sent_count)
                    new_doc = ""
                current_category = line[8:-8].strip()
            else:
                new_doc += (" "+line)
        if new_doc != "": # notes under the last category
            tmp_split_doc = split_doc(new_doc)
            tmp_sent_count = len(tmp_split_doc)
            sentences += tmp_split_doc
            categories += ([current_category] * tmp_sent_count)
            new_doc = ""
    sentences = sentences[:min(n_max_sentence_num, len(sentences))]
    categories = categories[:min(n_max_sentence_num, len(sentences))]
    return sentences, categories

# tokenize
def tokenize(text):
    tokenizer = re.compile('\w+|\*\*|[^\s\w]')
    return tokenizer.findall(text.lower())
# clean digits: 698 => 600
def clean_token(s):
    if len(s) > 1:
        if s.isdigit():
            l = len(s)
            s = str(int(s)//(10**(l-1)) * 10**(l-1))
    return s.lower()

# transfer category into index
def find_category_to_id(categories_per_file):
    categories_id_per_file = []
    for c in categories_per_file:
        categories_id_per_file.append(category_id[c])
    return categories_id_per_file

# generate x embedding from each file
def generate_token_embedding(file_name,mimic3_embedding):
    n_max_sentence_num = 1000 # truncated to 1000 sentences a document
    n_max_word_num = 25 # truncated to 25 words a sentence
    sentences_per_file, categories_per_file = generate_sent_category(file_name)

    # category id list
    categories_id_per_file = find_category_to_id(categories_per_file)
    number_of_sentences = len(sentences_per_file)
    categories_id_per_file = categories_id_per_file + [0]*(n_max_sentence_num-number_of_sentences) # padding zero to category id list

    # x_token
    x_token = []
    for i, sent in enumerate(sentences_per_file):
        x_sentence = []
        tokens = tokenize(sent)
        if len(tokens) == 0: # ignore empty token
            continue
        tokens_truncated = tokens[:min(n_max_word_num, len(tokens))]
        for j, tok in enumerate(tokens_truncated):
            tok = clean_token(tok)
            if tok in mimic3_embedding:
                x_sentence.append(mimic3_embedding[tok])
            else:
                x_sentence.append(mimic3_embedding['UNK'])
        x_sentence = np.stack(x_sentence)
        x_sentence = np.pad(x_sentence, ((0, n_max_word_num - x_sentence.shape[0]), (0, 0)), "constant")
        x_token.append(x_sentence)
    x_token = np.stack(x_token)
    x_token = np.pad(x_token, ((0,n_max_sentence_num - x_token.shape[0]),(0,0),(0,0)), "constant")

    return x_token, number_of_sentences, categories_id_per_file


# split train test dev
def split_train_test_dev(index_list,load_path, load=False):
    if not load:
        np.random.shuffle(index_list)
        n_dev = len(index_list) // 10
        dev_index = index_list[:n_dev]
        test_index = index_list[n_dev:(2*n_dev)]
        train_index = index_list[(2*n_dev):]
        np.save(load_path + '/dev.npy', dev_index)
        np.save(load_path + '/test.npy', test_index)
        np.save(load_path + '/train.npy', train_index)
    else:
        dev_index = np.load(load_path + "/dev.npy")
        train_index = np.load(load_path + "/train.npy")
        test_index = np.load(load_path + "/test.npy")
    return train_index,test_index,dev_index

def generate_label_from_dead_date(y_series,dead_date):
    label = []
    for index,y in y_series.iteritems():
        if y < dead_date:
            label.append([0,1])
        else:
            label.append([1,0])
    label = np.asarray(label)
    return label

# CNN model architecture
def CNN_model(input_x,input_ys, sent_length, category_index, dropout_keep_prob):
    # general config
    embedding_size = 100
    max_document_length = 1000
    max_sentence_length = 25
    filter_sizes = (3, 4, 5)
    num_filters = 50
    n_category = 15
    dim_category = 10
    document_filter_size = 3
    document_num_filters = 50

    num_classes = 2
    lambda_regularizer_strength = 5

    # category lookup
    target_embeddings = tf.get_variable(
                        name="target_embeddings",
                        dtype=tf.float32,
                        shape=[n_category, dim_category])
    embedded_category = tf.nn.embedding_lookup(target_embeddings,
                        category_index, name="target_embeddings") # [n_batch, n_doc,dim_category]

    # =============================== reshape to do word level CNN ============================================================================
    x = tf.reshape(input_x, [-1, max_sentence_length, embedding_size])
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv1d(
                value=x,
                filters=W,
                stride=1,
                padding="VALID"
            ) # shape: (n_batch*n_doc) * (n_seq - filter_size) * num_filters
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu") # shape not change

            # Maxpooling over the outputs
            # another implementation of max-pool
            pooled = tf.reduce_max(h, axis=1) # (n_batch*n_doc) * n_filter
            pooled_outputs.append(pooled) # three list of pooled array
    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 1) # shape: (n_batch*n_doc) * num_filters_total
    # Add dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool, dropout_keep_prob) # (n_batch * n_doc) * num_filters_total

    first_cnn_output = tf.reshape(h_drop, [-1, max_document_length, num_filters_total]) # [n_batch, n_doc, n_filter]
    first_cnn_output = tf.concat([first_cnn_output, embedded_category], axis=2) # [n_batch, n_doc, n_filter + dim_category]
    h_drop = tf.reshape(first_cnn_output,[-1, (num_filters_total+dim_category)]) # [(n_batch * n_doc), n_filter + dim_category]

#do sentence loss with the matrix of the concat result of category & h_drop
    total_loss = 0
    for (M,input_y) in enumerate(input_ys):
        with tf.name_scope("task"+str(M)):
            W = tf.Variable(tf.truncated_normal([(num_filters_total+dim_category), num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            scores_sentence = tf.nn.xw_plus_b(h_drop, W, b)
            # scores has shape: [(n_batch * n_doc), num_classes]

            # input_y: shape: [n_batch, num_classes]  have to transfer the same shape to scores
            y = tf.tile(input_y, [1, max_document_length])  #y: shape [n_batch, (num_classes*n_doc)]
            y = tf.reshape(y, [-1,num_classes]) #y: shape [(n_batch*n_doc), num_classes]

            sentence_losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores_sentence, labels=y)
            # sentence losses has shape: [(n_batch * n_doc), ] it is a 1D vector.
            sentence_losses = tf.reshape(sentence_losses, [-1, max_document_length]) # [n_batch, n_doc]
            mask = tf.sequence_mask(sent_length)
            sentence_losses = tf.boolean_mask(sentence_losses, mask)
            sentence_losses_avg = tf.reduce_mean(sentence_losses)
            total_loss += sentence_losses_avg * lambda_regularizer_strength

#===========================================sentence-level CNN ==================================================================
    filter_shape = [document_filter_size, (num_filters_total + dim_category), document_num_filters]
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[document_num_filters]), name="b")
    conv = tf.nn.conv1d(
        value=first_cnn_output,
        filters=W,
        stride=1,
        padding="VALID"
    ) # n_batch * (n_max_doc - filter+1) * doc_num_filters
    # Apply nonlinearity
    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
    # Maxpooling over the outputs
    # another implementation of max-pool
    pooled_second = tf.reduce_max(h, axis=1) # n_batch * document_num_filters
    with tf.name_scope("dropout"):
        pooled_second_drop = tf.nn.dropout(pooled_second, dropout_keep_prob)

    scores_soft_max_list = []
    for (M,input_y) in enumerate(input_ys):
        with tf.name_scope("task"+str(M)):
            W = tf.Variable(tf.truncated_normal([document_num_filters, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            scores = tf.nn.xw_plus_b(pooled_second_drop, W, b)
            # scores has shape: [n_batch, num_classes]
            scores_soft_max = tf.nn.softmax(scores)
            scores_soft_max_list.append(scores_soft_max)  # scores_soft_max_list shape:[multi_size, n_batch, num_classes]
            # predictions = tf.argmax(scores, axis=1, name="predictions")
            # predictions has shape: [None, ]. A shape of [x, ] means a vector of size x
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
            # losses has shape: [None, ]
            # include target replication
            # total_loss += losses
            loss_avg = tf.reduce_mean(losses)
            total_loss += loss_avg
    # avg_loss = tf.reduce_mean(total_loss)
    # optimize function
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    optimize = optimizer.minimize(total_loss)


    #correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
    #accuracy = tf.reduce_sum(tf.cast(correct_predictions, "float"), name="accuracy")

    return optimize, scores_soft_max_list


# evaluation

def evaluation(predictions, y_label):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(y_label)):
        if predictions[i]==y_label[i]== 1:
            tp += 1
        if predictions[i]==y_label[i]== 0:
            tn += 1
        if predictions[i]== 1 and y_label[i] == 0:
            fp += 1
        if predictions[i] == 0 and y_label[i] == 1:
            fn += 1
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall/(precision+recall)
    acc = (tp+tn)/(tp+tn+fp+fn)
    return acc
