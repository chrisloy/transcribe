import persist
import sys
import tensorflow as tf


if __name__ == '__main__':
    with tf.Session() as sess:
        persist.load(sess, sys.argv[1])
    print "Done"
