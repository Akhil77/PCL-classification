from simpletransformers.classification import ClassificationModel, ClassificationArgs, MultiLabelClassificationModel, MultiLabelClassificationArgs
from urllib import request
import pandas as pd
import logging
import torch
from collections import Counter
from ast import literal_eval
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

# prepare logger
logging.basicConfig(level=logging.INFO)

transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# check gpu
cuda_available = torch.cuda.is_available()
print('Cuda available? ',cuda_available)

# Get the GPU device name.
device_name = tf.test.gpu_device_name()
# The device name should look like the following:
if device_name == '/device:GPU:0':
  print('Found GPU at: {}'.format(device_name))
else:
  raise SystemError('GPU device not found')

df = pd.read_csv('dontpatronizeme_pcl.tsv', sep='\t', header=None)
df = df.drop(columns = [0])
df.columns = ['ID', 'keyword', 'country_code', 'paragraph', 'label']

def smush_labels(label):
  if label > 1:
    return 1
  else:
    return 0

df['label'] = df['label'].apply(smush_labels)
trdf1,tedf1 = train_test_split(df, test_size = 0.10, shuffle = True)
pcldf = trdf1[trdf1.label==1]
npos = len(pcldf)

training_set1 = pd.concat([pcldf,trdf1[trdf1.label==0][:int(npos*2.5)]])
print(len(training_set1))

task1_model_args = ClassificationArgs(num_train_epochs=10, 
                                      no_save=True, 
                                      no_cache=True, 
                                      overwrite_output_dir=True,
                                     train_batch_size=32,
                                     save_steps = -1,
                                     save_model_every_epoch = False,
                                     #use_early_stopping = True,
                                     early_stopping_delta = 0.01,
                                     early_stopping_metric = "mcc",
                                     early_stopping_metric_minimize = False,
                                     early_stopping_patience = 3,
                                     #evaluate_during_training = True,
                                     evaluate_during_training_steps = 150,
                                     #evaluate_during_training_verbose = True,
                                     use_cached_eval_features= True)

task1_model = ClassificationModel("xlnet", 
                                  'xlnet-base-cased', 
                                  args = task1_model_args, 
                                  num_labels=2,
                                  weight=[0.25,0.65],
                                  use_cuda=cuda_available)

task1_model.train_model(training_set1[['paragraph', 'label']], eval_df=tedf1[['paragraph', 'label']])

preds_task1, _ = task1_model.predict(tedf1.paragraph.tolist())

preds_task1_bert = [[k] for k in preds_task1]
def labels2file(p, outf_path):
	with open(outf_path,'w') as outf:
		for pi in p:
			outf.write(','.join([str(k) for k in pi])+'\n')
labels2file(preds_task1_bert, os.path.join('res/', 'task1.txt'))
labels2file([[k] for k in tedf1.label.values], os.path.join('ref/', 'task1.txt'))

###########################################################################Organizers test train split ############

module_url = f"https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/master/semeval-2022/dont_patronize_me.py"
module_name = module_url.split('/')[-1]
print(f'Fetching {module_url}')
#with open("file_1.txt") as f1, open("file_2.txt") as f2
with request.urlopen(module_url) as f, open(module_name,'w') as outf:
  a = f.read()
  outf.write(a.decode('utf-8'))

from dont_patronize_me1 import DontPatronizeMe

dpm = DontPatronizeMe('.', '.')
dpm.load_task1()
dpm.load_task2(return_one_hot=True)

trids = pd.read_csv('train_semeval_parids-labels.csv')
teids = pd.read_csv('dev_semeval_parids-labels.csv')
trids.par_id = trids.par_id.astype(str)
teids.par_id = teids.par_id.astype(str)

rows = [] # will contain par_id, label and text
for idx in range(len(trids)):  
  parid = trids.par_id[idx]
  #print(parid)
  # select row from original dataset to retrieve `text` and binary label
  text = dpm.train_task1_df.loc[dpm.train_task1_df.par_id == parid].text.values[0]
  label = dpm.train_task1_df.loc[dpm.train_task1_df.par_id == parid].label.values[0]
  rows.append({
      'par_id':parid,
      'text':text,
      'label':label
  })
trdf2 = pd.DataFrame(rows)

rows = [] # will contain par_id, label and text
for idx in range(len(teids)):  
  parid = teids.par_id[idx]
  #print(parid)
  # select row from original dataset
  text = dpm.train_task1_df.loc[dpm.train_task1_df.par_id == parid].text.values[0]
  label = dpm.train_task1_df.loc[dpm.train_task1_df.par_id == parid].label.values[0]
  rows.append({
      'par_id':parid,
      'text':text,
      'label':label
  })
tedf2 = pd.DataFrame(rows)

# downsample negative instances
pcldf = trdf2[trdf2.label==1]
npos = len(pcldf)

training_set2 = pd.concat([pcldf,trdf2[trdf2.label==0][:int(npos*2.5)]])

task1_model_args = ClassificationArgs(num_train_epochs=5, 
                                      no_save=True, 
                                      no_cache=True, 
                                      overwrite_output_dir=True,
                                     train_batch_size=32,
                                     #evaluate_during_training = True,
                                     #evaluate_during_training_steps = 86,
                                     evaluate_during_training_verbose = True,
                                     use_cached_eval_features = True,
                                     #use_early_stopping = True,
                                     early_stopping_delta = 0.01,
                                     early_stopping_metric = "mcc",
                                     early_stopping_metric_minimize = True,
                                     early_stopping_patience = 2)

task1_model = ClassificationModel("xlnet", 
                                  'xlnet-base-cased', 
                                  args = task1_model_args, 
                                  num_labels=2,
                                  weight=[0.25,0.70],
                                  use_cuda=cuda_available)

task1_model.train_model(training_set2[['text', 'label']], eval_df=tedf2[['text', 'label']])

preds_task1,raw = task1_model.predict(tedf2.text.tolist())

preds_task1_bert = [[k] for k in preds_task1]
labels2file(preds_task1_bert, os.path.join('res/', 'task1.txt'))
labels2file([[k] for k in tedf2.label.values], os.path.join('ref/', 'task1.txt'))