from domain import Params
import time
import train


def grid_search_frame_models():

    hidden_nodes = [[], [88], [176], [176, 132]]
    learning_rate = [0.0001, 0.001, 0.01, 0.1]
    epochs = [10, 100]
    batch_size = 512
    dropout = [None, 0.8]

    params = []

    for hn in hidden_nodes:
        for lr in learning_rate:
            for e in epochs:
                for d in dropout:
                    params.append(
                        Params(
                            epochs=100,
                            train_size=600,
                            test_size=200,
                            corpus="16k_piano_notes_88_poly_3_to_15_velocity_63_to_127",
                            batch_size=batch_size,
                            hidden_nodes=hn,
                            dropout=None,
                            learning_rate=lr,
                        )
                    )

    print "Produced %d parameter configurations" % len(params)

    data = train.load_data(params[0], from_cache=True)

    for i, p in enumerate(params):
        before = time.time()
        print "RUN:", p.__dict__
        name, error = train.run_frame_model(p, d=data, ui=False, log=False)
        dur = time.time() - before
        print "Completed run [%02d/%d] graph [%s] error [%0.8f] time [%0.4f]" % (i+1, len(params), name, error, dur)


if __name__ == '__main__':
    grid_search_frame_models()