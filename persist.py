import data
import domain
import json
import model
import namer
import tensorflow as tf


def save(sess, m, d, p):
    graph_id = namer.new_name()
    print "Saving variables..."
    saver = tf.train.Saver()
    saver.save(sess, "graphs/%s-variables.ckpt" % graph_id)
    results = {
        "train_err": float(m.report_target.eval(feed_dict=m.train_labelled_feed(d))),
        "test_err": float(m.report_target.eval(feed_dict=m.test_labelled_feed(d)))
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
    m = None
    if p.graph_type == 'mlp':
        m = model.feed_forward_model(
            660,
            p.outputs(),
            hidden_nodes=p.hidden_nodes,
            learning_rate=p.learning_rate)
    else:
        assert 2 + 2 == 5, "Unsupported graph type %s" % p.graph_type
    saver = tf.train.Saver()
    saver.restore(sess, "graphs/%s-variables.ckpt" % graph_id)
    return m, p
