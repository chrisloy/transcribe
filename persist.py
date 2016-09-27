from collections import defaultdict
import domain
import json
import model
import namer
import tensorflow as tf


def save(sess, m, d, p, threshold=0.5):
    graph_id = namer.new_name()
    print "Saving variables..."
    saver = tf.train.Saver()
    saver.save(sess, "graphs/%s-variables.ckpt" % graph_id)
    results = {
        "dev_err": float(m.report_target.eval(feed_dict=m.dev_labelled_feed(d))),
        "test_err": float(m.report_target.eval(feed_dict=m.test_labelled_feed(d))),
        "threshold": threshold
    }
    with open('graphs/%s-meta.json' % graph_id, 'w') as outfile:
        json.dump({"params": p.__dict__, "results": results}, outfile)
    print "Saved graph %s" % graph_id
    return graph_id


def load(sess, graph_id, features=660):
    with open('graphs/%s-meta.json' % graph_id, 'r') as infile:
        dx = json.load(infile)
        params = dx["params"]
    p = domain.Params(**params)
    m = None
    if p.sequence_learning_rate:
        _, m = model.hierarchical_deep_network(
            features,
            p.outputs(),
            p.steps,
            p.frame_hidden_nodes,
            p.frame_dropout,
            p.frame_learning_rate,
            p.sequence_hidden_nodes,
            p.sequence_dropout,
            p.sequence_learning_rate)
    elif p.graph_type == 'mlp':
        m = model.feed_forward_model(
            features,
            p.outputs(),
            hidden_nodes=p.hidden_nodes,
            learning_rate=p.learning_rate)
    elif p.graph_type == 'ladder':
        m = model.ladder_model(
            features,
            p.outputs(),
            p.learning_rate,
            hidden_nodes=p.hidden_nodes
        )
    else:
        assert 2 + 2 == 5, "Unsupported graph type %s" % p.graph_type
    saver = tf.train.Saver()
    saver.restore(sess, "graphs/%s-variables.ckpt" % graph_id)
    return m, p
