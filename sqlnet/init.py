from default_args import Args
import torch
from sqlnet.utils import (
    load_dataset,
    load_word_emb,
    best_model_name,
)
from sqlnet.model.sqlnet import SQLNet
import spacy
import numpy as np
from sqlnet.model.modules.net_utils import run_lstm


args = Args(ca=True)
USE_SMALL = False
GPU = True
BATCH_SIZE = 64
TRAIN_ENTRY = (True, True, True)  # (AGG, SEL, COND)
TRAIN_AGG, TRAIN_SEL, TRAIN_COND = TRAIN_ENTRY
learning_rate = 1e-4 if args.rl else 1e-3

sql_data, table_data, val_sql_data, val_table_data, test_sql_data, \
    test_table_data, TRAIN_DB, DEV_DB, TEST_DB = load_dataset(
        args.dataset, use_small=USE_SMALL
    )

word_emb = load_word_emb(
    "glove/glove.%dB.%dd.npy" % (42, 300),
    load_used=args.train_emb,
    use_small=USE_SMALL,
)

model = SQLNet(
    word_emb,
    N_word=300,
    use_ca=args.ca,
    gpu=GPU,
    trainable_emb=args.train_emb)

agg_m, sel_m, cond_m = best_model_name(args, for_load=True)
model.agg_pred.load_state_dict(torch.load(agg_m))
model.sel_pred.load_state_dict(torch.load(sel_m))
model.cond_pred.load_state_dict(torch.load(cond_m))
_ = model.eval()


nlp = spacy.load('en_core_web_sm')


def process_query(q):
    tokens = [c.text.lower() for c in nlp(q)]
    zv = np.zeros(model.embed_layer.N_word, dtype=np.float32)
    X = np.zeros((len(tokens) + 2, model.embed_layer.N_word), dtype=np.float32)
    for i, t in enumerate(tokens):
        X[i + 1] = model.embed_layer.word_emb.get(t, zv.copy())
    var_inp = torch.from_numpy(X)
    if model.gpu:
        var_inp = var_inp.cuda()
    return torch.autograd.Variable(var_inp)


text = "What position does the player who played for butler cc (ks) play?"
q = process_query(text)

# Get the agg predictor working.
h_enc, _ = run_lstm(model.agg_pred.agg_lstm, q.reshape((1,) + q.shape), np.array([q.shape[0]]))


from IPython import embed; embed()  # NOQA
