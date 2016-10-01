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
    test_error = m.report_target.eval(feed_dict=m.test_labelled_feed(d))
    results = {
        "dev_err": float(m.report_target.eval(feed_dict=m.dev_labelled_feed(d))),
        "test_err": float(test_error),
        "threshold": threshold
    }
    with open('graphs/%s-meta.json' % graph_id, 'w') as outfile:
        json.dump({"params": p.__dict__, "results": results}, outfile)
    print "Saved graph %s" % graph_id
    return graph_id, test_error


def load(sess, graph_id, features=660):
    with open('graphs/%s-meta.json' % graph_id, 'r') as infile:
        dx = json.load(infile)
        params = dx["params"]
        threshold = dx["results"]["threshold"]
    p = domain.Params(**params)
    if p.graph_type == 'mlp_mlp':
        _, m, _ = model.hierarchical_deep_network(
            features,
            p.outputs(),
            p.steps,
            p.frame_hidden_nodes,
            p.frame_dropout,
            p.frame_learning_rate,
            p.sequence_hidden_nodes,
            p.sequence_dropout,
            p.sequence_learning_rate)
    elif p.graph_type == 'mlp_rnn':
        _, m, _ = model.hierarchical_recurrent_network(
            features,
            p.outputs(),
            p.steps,
            p.frame_hidden_nodes,
            p.frame_dropout,
            p.frame_learning_rate,
            p.rnn_graph_type,
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
            p.hidden_nodes,
            p.noise_var,
            p.noise_costs
        )
    else:
        assert False, "Unsupported graph type %s" % p.graph_type
    saver = tf.train.Saver()
    saver.restore(sess, "graphs/%s-variables.ckpt" % graph_id)
    return m, p, threshold
