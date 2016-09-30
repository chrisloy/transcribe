from domain import Params
import time
import train


def grid_search_frame_models():

    graph_type = ['lstm', 'rnn', 'bi_rnn', 'bi_lstm']
    learning_rate = [0.00001, 0.0001, 0.001]
    steps = [32, 64]

    params = []

    for gt in graph_type:
        for lr in learning_rate:
            for s in steps:
                params.append(
                    Params(
                        epochs=250,
                        train_size=600,
                        test_size=200,
                        corpus="16k_piano_notes_88_poly_3_to_15_velocity_63_to_127",
                        batch_size=(512 / s),
                        graph_type='mlp_rnn',
                        # Inspired by burt-hankies
                        frame_dropout=None,
                        frame_epochs=250,
                        frame_hidden_nodes=[44],
                        frame_learning_rate=0.003,
                        steps=s,
                        rnn_graph_type=gt,
                        sequence_learning_rate=lr
                    )
                )

    print "Produced %d parameter configurations" % len(params)

    for i, p in enumerate(params):
        from tensorflow.python.framework import ops
        ops.reset_default_graph()
        before = time.time()
        name, error = train.run_hierarchical_model(p, ui=False, report_epochs=100)
        dur = time.time() - before
        print "Completed run [%02d/%d] graph [%s] error [%0.8f] time [%0.4f]" % (i+1, len(params), name, error, dur)


if __name__ == '__main__':
    grid_search_frame_models()