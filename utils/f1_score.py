import keras.backend as K
import tensorflow as tf
import numpy as np

def base_f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return f1

def f1_min(y_true, y_pred):
    f1 = base_f1(y_true, y_pred)
    return K.min(f1)

def f1_max(y_true, y_pred):
    f1 = base_f1(y_true, y_pred)
    return K.max(f1)

def f1_mean(y_true, y_pred):
    f1 = base_f1(y_true, y_pred)
    return K.mean(f1)

def f1_std(y_true, y_pred):
    f1 = base_f1(y_true, y_pred)
    return K.std(f1)

def get_scores(partitions, predictions, train_labels):
    scores = []
    with tf.Session() as sess:
        for partition,prediction in zip(partitions,predictions):
            validation_labels = train_labels.loc[train_labels.Id.isin(partition["validation"])].copy()
            y_true = validation_labels.drop(["Id", "Target"], axis=1).values
            y_pred = np.where(prediction.values > 0.5, 1, 0)
            scores.append(sess.run(base_f1(y_true.flatten(), y_pred.flatten())))
    
    return scores