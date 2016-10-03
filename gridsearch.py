from domain import Params
import data
import time
import train


def grid_search_frame_models():

    graph_type = ['mlp']
    learning_rate = [0.001, 0.01]
    epochs = [50, 100, 200]
    hidden = [[88], [66], [44]]

    batch_size = 512

    params = []

    for gt in graph_type:
        for lr in learning_rate:
            for e in epochs:
                for h in hidden:
                    params.append(
                        Params(
                            epochs=e,
                            batch_size=batch_size,
                            graph_type=gt,
                            hidden_nodes=h,
                            dropout=None,
                            learning_rate=lr,
                        )
                    )

    print "Produced %d parameter configurations" % len(params)

    d = data.maps_cross_instruments(batch_size)

    results = []

    for i, p in enumerate(params):
        from tensorflow.python.framework import ops
        ops.reset_default_graph()
        before = time.time()
        name, error, f1 = train.run_frame_model(p, ui=False, report_epochs=1, d=d)
        dur = time.time() - before
        results.append((name, error, f1))
        print "Completed run [%02d/%d] graph [%s] error [%0.8f] time [%0.4f]" % (i+1, len(params), name, error, dur)

    print "SUMMARY:"
    for n, e, f in results:
        print "[%s] : Test ---> Error: [%0.8f] ---> F1: [%0.8f]" % n, e, f


def grid_search_sequence_models():

    graph_type = ['bi_rnn', 'rnn']
    learning_rate = [0.0001, 0.001, 0.01]
    epochs = [500]
    hidden = [[88], [66], [44], [22]]

    steps = 16
    batch_size = 512 / steps

    params = []

    for lr in learning_rate:
        for gt in graph_type:
            for e in epochs:
                for h in hidden:
                    params.append(
                        Params(
                            epochs=e,
                            batch_size=batch_size,
                            graph_type='mlp_rnn',
                            frame_hidden_nodes=h,
                            frame_dropout=None,
                            frame_learning_rate=0.01,  # not used
                            steps=steps,
                            rnn_graph_type=gt,
                            sequence_learning_rate=lr
                        )
                    )

    print "Produced %d parameter configurations" % len(params)

    # d = data.maps_cross_instruments(batch_size).to_sequences(steps)
    d = data.load(600, 200, 512, True, batch_size, "16k_piano_notes_88_poly_3_to_15_velocity_63_to_127", 21, 109)
    d = d.to_sequences(steps)

    results = []

    for i, p in enumerate(params):
        from tensorflow.python.framework import ops
        ops.reset_default_graph()
        before = time.time()
        name, error, f1 = train.run_hierarchical_model(p, ui=False, report_epochs=10, d=d)
        dur = time.time() - before
        results.append((name, error, f1))
        print "Completed run [%02d/%d] graph [%s] error [%0.8f] time [%0.4f]" % (i+1, len(params), name, error, dur)

    print "SUMMARY:"
    for n, e, f in results:
        print "[%s] : Test ---> Error: [%0.8f] ---> F1: [%0.8f]" % n, e, f

if __name__ == '__main__':
    grid_search_sequence_models()