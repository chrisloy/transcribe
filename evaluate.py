import sys
import tensorflow as tf
from tensorflow.python.platform import gfile


def restore(graph_id):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, graph_id)
        print("load graph")
        with gfile.FastGFile("graphs/%s-graph.pbtxt" % graph_id,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        print("map variables")
        persisted_result = sess.graph.get_tensor_by_name("saved_result:0")
        tf.add_to_collection(tf.GraphKeys.VARIABLES, persisted_result)
        try:
            saver = tf.train.Saver(tf.all_variables())
        except:
            pass
        print("load data")
        saver.restore(sess, "checkpoint.data")  # now OK
        print(persisted_result.eval())
        print("DONE")


if __name__ == '__main__':
    graph_id = sys.argv[1]
    restore(graph_id)