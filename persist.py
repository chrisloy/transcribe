import data
import domain
import json
import model
import uuid
import tensorflow as tf


def save(sess, m, d, p):
    graph_id = str(uuid.uuid4())
    print "Saving variables..."
    saver = tf.train.Saver()
    saver.save(sess, "graphs/%s-variables.ckpt" % graph_id)
    results = {
        "train_err": float(m.report_target.eval(feed_dict={m.x: d.x_train, m.y_gold: d.y_train})),
        "test_err": float(m.report_target.eval(feed_dict={m.x: d.x_test, m.y_gold: d.y_test}))
    }
    with open('graphs/%s-meta.json' % graph_id, 'w') as outfile:
        json.dump({"params": p.to_dict(), "results": results}, outfile)
    print "Saved graph %s" % graph_id
    return graph_id


def load(sess, graph_id):
    with open('graphs/%s-meta.json' % graph_id, 'r') as infile:
        dx = json.load(infile)
        params = dx["params"]
    p = domain.from_dict(params)
    d = data.load(p.train_size, p.test_size, p.slice_samples, True, p.batch_size, p.corpus).to_one_hot()
    m = model.feed_forward_model(
        d.features,
        89,
        hidden_nodes=p.hidden_nodes,
        loss_function="cross_entropy",
        learning_rate=0.02)
    saver = tf.train.Saver()
    saver.restore(sess, "graphs/%s-variables.ckpt" % graph_id)
    return m, d, p
