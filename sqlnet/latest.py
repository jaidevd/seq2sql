# coding: utf-8
N_word, B_word = 42
N_word, B_word = 300, 42
USE_SMALL = False
GPU = True
BATCH_SIZE = 64
agg, sel, cond = True, True, True
from sqlnet.utils import load_dataset
sql_data, table_data, val_sql_data, val_table_data, test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB = load_dataset(
0, use_small=False)
type(sql_data)
len(sql_data)
sql_data[0]
sql_data[0]['table_id']
table_data
get_ipython().run_line_magic('clear', '')
type(table_data)
table_data.keys()
table_data[sql_data[0]['table_id']]
len(table_data)
type(TRAIN_DB)
TRAIN_DB
from sqlnet.utils import load_word_emb
word_emb = load_word_emb('glove/glove.%dB.%dd.txt' % (B_word, N_word), load_uused=True, use_small=False)
word_emb = load_word_emb('glove/glove.%dB.%dd.txt' % (B_word, N_word), load_used=True, use_small=False)
word_emb = load_word_emb('glove/glove.%dB.%dd.txt' % (B_word, N_word), load_used=False, use_small=False)
type(word_emb)
len(word_emb)
for k, v in word_emb.items():
    break
    
k
v
v.shape
len(word_emb)
from sqlnet.model.sqlnet import SQLNet
model = SQLNet(word_emb, N_word=N_word, use_ca=True, gpu=True, trainable_emb=False)
model
get_ipython().run_line_magic('pinfo', 'model.load_state_dict')
