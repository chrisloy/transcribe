from domain import Params
import time
import train


def grid_search_frame_models():

    graph_type = ['ladder']
    hidden_nodes = [[88], [88, 88], [176], [176, 132]]
    learning_rate = [0.0001, 0.001, 0.01, 0.1]
    batch_size = [512, 1024]
    dropout = [None, 0.8]

    params = []

    for gt in graph_type:
        for bs in batch_size:
            for hn in hidden_nodes:
                for lr in learning_rate:
                    for d in dropout:
                        params.append(
                            Params(
                                epochs=500,
                                train_size=600,
                                test_size=200,
                                corpus="16k_piano_notes_88_poly_3_to_15_velocity_63_to_127",
                                batch_size=bs,
                                hidden_nodes=hn,
                                dropout=d,
                                learning_rate=lr,
                                graph_type=gt
                            )
                        )

    print "Produced %d parameter configurations" % len(params)

    data = train.load_data(params[0], from_cache=True)

    for i, p in enumerate(params):
        before = time.time()
        name, error = train.run_frame_model(p, d=data, ui=False, log=False, early_stop=True)
        dur = time.time() - before
        print "Completed run [%02d/%d] graph [%s] error [%0.8f] time [%0.4f]" % (i+1, len(params), name, error, dur)


if __name__ == '__main__':
    grid_search_frame_models()