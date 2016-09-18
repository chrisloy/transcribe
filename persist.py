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
    feed_train = {m.x: d.x_train, m.y_gold: d.y_train}
    feed_test = {m.x: d.x_test, m.y_gold: d.y_test}
    results = {
        "train_err": float(m.report_target.eval(feed_dict=feed_train)),
        "test_err": float(m.report_target.eval(feed_dict=feed_test))
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
    m = model.feed_forward_model(
        660,
        p.outputs(),
        hidden_nodes=p.hidden_nodes,
        learning_rate=p.learning_rate)
    saver = tf.train.Saver()
    saver.restore(sess, "graphs/%s-variables.ckpt" % graph_id)
    return m, p
