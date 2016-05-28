import sys
sys.path.append('../')
import cPickle as pickle
import re
import glob
import os
import matplotlib.pyplot as plt
import time

import theano
import theano.tensor as T
import numpy as np
import pandas as p
import lasagne as nn

from utils import hms, architecture_string, get_img_ids_from_iter
dump_path = 'dumps/2015_07_17_123003.pkl'
model_data = pickle.load(open(dump_path, 'r'))
l_out = model_data['l_out']
l_ins = model_data['l_ins']
model_arch = architecture_string(model_data['l_out'])

num_params = nn.layers.count_params(l_out)
model_arch += "\nNumber of parameters: %d.\n\n" % num_params

# Get some training/validation info.
selected_keys = ['acc_eval_train', 'acc_eval_valid',
                 'losses_eval_train', 'losses_eval_valid',
                 'metric_eval_train', 'metric_eval_valid',
                 'metric_cont_eval_train', 'metric_cont_eval_valid']
model_metrics = {key: model_data[key]
                 for key in selected_keys if key in model_data}

res_df = p.DataFrame(model_metrics)
#print res_df
model_arch += 'BEST/LAST KAPPA TRAIN: %.3f - %.3f.\n' % (
    res_df.metric_eval_train.max(),
    res_df.metric_eval_train.iloc[-1]
)
model_arch += 'BEST/LAST KAPPA VALID: %.3f - %.3f.\n' % (
    res_df.metric_eval_valid.max(),
    res_df.metric_eval_valid.iloc[-1]
)

model_arch += '\nBEST/LAST ACC TRAIN: %.2f - %.2f.\n' % (
    res_df.acc_eval_train.max() * 100,
    res_df.acc_eval_train.iloc[-1] * 100
)

model_arch += 'BEST/LAST ACC VALID: %.2f - %.2f.\n' % (
    res_df.acc_eval_valid.max() * 100,
    res_df.acc_eval_valid.iloc[-1] * 100
)

model_arch += '\nTOTAL TRAINING TIME: %s' % \
              hms(model_data['time_since_start'])

#print model_arch
train_conf_mat, hist_rater_a, \
        hist_rater_b, train_nom, \
        train_denom = model_data['metric_extra_eval_train'][-1]
valid_conf_mat, hist_rater_a, \
        hist_rater_b, valid_nom, \
        valid_denom = model_data['metric_extra_eval_valid'][-1]
# Normalised train confusion matrix (with argmax decoding).
#print train_conf_mat / train_conf_mat.sum()
# Normalised validation confusion matrix (with argmax decoding).
#print valid_conf_mat / valid_conf_mat.sum()
chunk_size = model_data['chunk_size'] * 2
batch_size = model_data['batch_size']

print "Batch size: %i." % batch_size
print "Chunk size: %i." % chunk_size
output = nn.layers.get_output(l_out, deterministic=True)
input_ndims = [len(nn.layers.get_output_shape(l_in))
               for l_in in l_ins]
xs_shared = [nn.utils.shared_empty(dim=ndim)
             for ndim in input_ndims]
idx = T.lscalar('idx')

givens = {}
for l_in, x_shared in zip(l_ins, xs_shared):
    givens[l_in.input_var] = x_shared[idx * batch_size:(idx + 1) * batch_size]

compute_output = theano.function(
    [idx],
    output,
    givens=givens,
    on_unused_input='ignore'
)
# Do transformations per patient instead?
if 'paired_transfos' in model_data:
    paired_transfos = model_data['paired_transfos']
else:
    paired_transfos = False
    
#print paired_transfos
train_labels = p.read_csv(os.path.join('data/trainLabels.csv'))
#print train_labels.head(5)
# Get all patient ids.
patient_ids = sorted(set(get_img_ids_from_iter(train_labels.image)))
num_chunks = int(np.ceil((2 * len(patient_ids)) / float(chunk_size)))
# Where all the images are located: 
# it looks for [img_dir]/[patient_id]_[left or right].jpeg
#img_dir = '/storage/hpc_dmytro/Kaggle/DR/test/'
img_dir = '/storage/hpc_dmytro/Kaggle/DR/processed/run-stretch/test/'
#img_dir = '/storage/hpc_gagand87/train/jpeg/'
from generators import DataLoader
data_loader = DataLoader()
new_dataloader_params = model_data['data_loader_params']
new_dataloader_params.update({'images_test': patient_ids})
new_dataloader_params.update({'labels_test': train_labels.level.values})
new_dataloader_params.update({'prefix_train': img_dir})
data_loader.set_params(new_dataloader_params)
def do_pred(test_gen):
    outputs = []
    for e, (xs_chunk, chunk_shape, chunk_length) in enumerate(test_gen()):
        num_batches_chunk = int(np.ceil(chunk_length / float(batch_size)))

        print "Chunk %i/%i" % (e + 1, num_chunks)

        print "  load data onto GPU"
        for x_shared, x_chunk in zip(xs_shared, xs_chunk):
            x_shared.set_value(x_chunk)

        print "  compute output in batches"
        outputs_chunk = []
        for b in xrange(num_batches_chunk):
            out = compute_output(b)
            outputs_chunk.append(out)

        outputs_chunk = np.vstack(outputs_chunk)
        outputs_chunk = outputs_chunk[:chunk_length]

        outputs.append(outputs_chunk)

    return np.vstack(outputs), xs_chunk
no_transfo_params = model_data['data_loader_params']['no_transfo_params']

print no_transfo_params
# The default gen with "no transfos".
test_gen = lambda: data_loader.create_fixed_gen(
    data_loader.images_test[:],
    chunk_size=chunk_size,
    prefix_train=img_dir,
    prefix_test=img_dir,
    transfo_params=no_transfo_params,
    paired_transfos=paired_transfos,
)
from metrics import continuous_kappa
outputs_orig, chunk_orig = do_pred(test_gen)
outputs_labels = np.argmax(outputs_orig, axis=1)

print type(outputs_labels)
np.savetxt('output.csv', outputs_labels.astype(int))
'''
kappa_eval = continuous_kappa(
                outputs_labels,
                train_labels.level.values[:outputs_labels.shape[0]],
            )

metric, conf_mat, \
    hist_rater_a, hist_rater_b, \
    nom, denom = kappa_eval
    
print 'Kappa %.4f' % metric, '\n'
print conf_mat, '\n'
print nom, '\n'
print nom / nom.sum(), nom.sum()
train_imgs = set(data_loader.images_train_0)
valid_idx = [0  if img in train_imgs else 1 for img in data_loader.images_test]
df_preds = p.DataFrame([train_labels.image[:outputs_labels.shape[0]],
                        outputs_labels,
                        train_labels.level.values[:outputs_labels.shape[0]],
                       np.repeat(valid_idx, 2)[:outputs_labels.shape[0]]]).T
df_preds.columns = ['image', 'pred', 'true', 'valid']
print df_preds[df_preds.pred != df_preds.true]
diag_out = theano.function(
    [idx],
    nn.layers.get_output(nn.layers.get_all_layers(l_out), deterministic=True),
    givens=givens,
    on_unused_input="ignore"
)
diag_result = np.asarray(diag_out(0))
# The input images.
print diag_result[0].shape
def plot_rollaxis(im, figsize=(15, 15), 
                  zmuv_mean=data_loader.zmuv_mean, 
                  zmuv_std=data_loader.zmuv_std,
                 norm=True, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, figsize=figsize)
        
    if norm:
        ax.imshow((zmuv_std[0] + 0.05) * np.rollaxis(im, 0, 3) + zmuv_mean[0])
    else:
        ax.imshow(np.rollaxis(im, 0, 3))
        
    return ax
plot_rollaxis(diag_result[0][1])
df_chunk = df_preds[-128*2:]
df_chunk['idx'] = np.repeat(range(128), 2)
print df_chunk
# To print some output for a layer. (Hacky / quick.)
def print_output(layer_out, norm=False):
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(15, 200))

    for i, elem in enumerate(np.asarray(layer_out)[:2]):
        print elem.shape
        
        if norm:
            ax[i].imshow(np.concatenate(elem, axis=0), cmap=plt.cm.gray, 
                         vmin=np.asarray(layer_out).min(),
                         vmax=np.asarray(layer_out).max())
        else:        
            ax[i].imshow(np.concatenate(elem, axis=0), cmap=plt.cm.gray)
idx = 8
print df_chunk[df_chunk.idx == idx] 
print "bend"
plot_rollaxis(diag_result[0][2*idx+0])  # Left.
plot_rollaxis(diag_result[0][2*idx+1])  # Right.
print "bend"'''
