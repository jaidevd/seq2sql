# coding: utf-8
embedding.forward(q)
embedding(q)
get_ipython().run_line_magic('pinfo', 'embedding.get_x_batch')
embedding.get_x_batch
get_ipython().run_line_magic('pinfo', 'embedding.gen_x_batch')
get_ipython().run_line_magic('pinfo2', 'embedding.gen_x_batch')
embedding.word_emb
type(embedding.word_emb)
process_query(q)
q
get_ipython().run_line_magic('clear', '')
text
qq = process_query(text)
qq
qq.shape
text
model.SQL_TOK
len(_)
len(model.SQL_TOK)
18 + 8 - 2
agg_m
model.agg_pred.forward
model.agg_pred.agg_lstm
from sqlnet.model.modules.net_utils import run_lstm
run_lstm(model.agg_pred.agg_lstm, q, [17])
run_lstm(model.agg_pred.agg_lstm, q, 17)
run_lstm(model.agg_pred.agg_lstm, q, np.array([17]))
qq.shape
q
run_lstm(model.agg_pred.agg_lstm, q.reshape(1, 17, 300), np.array([17]))
h_enc, _ = run_lstm(model.agg_pred.agg_lstm, q.reshape(1, 17, 300), np.array([17]))
h_enc.shape
