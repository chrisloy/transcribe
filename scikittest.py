import data
import numpy as np
import train
from domain import Params
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from_cache = True

p = Params(
    epochs=100,
    train_size=80,
    test_size=20,
    hidden_nodes=[],
    corpus="five_piano_simple",
    learning_rate=0.5,
    lower=21,
    upper=109,
    padding=0
)

d = data.load(
    p.train_size, p.test_size, p.slice_samples, from_cache, p.batch_size, p.corpus, p.lower, p.upper
).to_padded(p.padding).shuffle_frames().to_sparse()


notes = np.arange(60, 65)
offset = notes[0]

for pen in ['l1', 'l2']:
    for icp in (True, False):

        print "PEN=%s, intercept=%s" % (pen, str(icp))

        y_pred = np.zeros((d.y_test.shape[0], len(notes)))

        for n in notes:
            print "Fitting note %d..." % n
            lr = LogisticRegression(penalty=pen, fit_intercept=icp)
            lr.fit(d.x_train, d.y_train[:, n])
            scores = lr.predict_proba(d.x_test)[:, 1]
            y_pred[:, n - offset] = scores

        train.report_poly_stats(y_pred, d.y_test[:, notes], show_graph=False)
