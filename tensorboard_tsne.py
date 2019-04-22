import os
import tensorflow as tf
import numpy as np
import HP
from tensorflow.contrib.tensorboard.plugins import projector

# run tsne in tensorboard

# patient vector as features
representation = np.load(HP.tsne_vector_directory)

# label for the vectors
metadata_path = HP.tsne_label_directory

# checkpoint log
LOG_DIR = HP.tensorboard_log

representation_var = tf.Variable(representation,name='representation')

with tf.Session() as sess:
    saver = tf.train.Saver([representation_var])

    sess.run(representation_var.initializer)

    saver.save(sess, os.path.join(LOG_DIR, "vector.ckpt"))

    config = projector.ProjectorConfig()

    embedding = config.embeddings.add()
    embedding.tensor_name = representation_var.name

    embedding.metadata_path = metadata_path

    summary_writer = tf.summary.FileWriter(LOG_DIR)

    projector.visualize_embeddings(summary_writer, config)
