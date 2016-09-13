from domain import Params
from train import run_one_hot_joint_model, run_frame_model, run_sequence_model


def run_best_time_slice(corpus):
    # TODO just search graphs for this
    if corpus == "two_piano_one_octave":
        # 0.14709036
        run_frame_model(
            Params(
                epochs=10,
                train_size=40,
                test_size=10,
                hidden_nodes=[],
                corpus="two_piano_one_octave",
                learning_rate=0.5,
                lower=60,
                upper=72,
                padding=0
            )
        )
    elif corpus == "mono_piano_one_octave":
        run_one_hot_joint_model(
            Params(
                epochs=100,
                train_size=40,
                test_size=10,
                hidden_nodes=[],
                corpus="mono_piano_one_octave",
                learning_rate=0.05,
                lower=60,
                upper=72,
                padding=0
            )
        )
    elif corpus == "mono_piano_two_octaves":
        run_one_hot_joint_model(
            Params(
                epochs=100,
                train_size=40,
                test_size=10,
                hidden_nodes=[],
                corpus="mono_piano_two_octaves",
                learning_rate=0.05,
                lower=48,
                upper=72,
                padding=0
            )
        )
    elif corpus == "five_piano_magic":
        # 0.09098092
        run_frame_model(
            Params(
                epochs=4,
                train_size=400,
                test_size=100,
                hidden_nodes=[],
                corpus="five_piano_magic",
                learning_rate=0.1,
                lower=21,
                upper=109,
                padding=0
            )
        )
    elif corpus == "piano_notes_88_poly_3_to_15_velocity_63_to_127":
        # 0.15406726  /  0.919213 ROC AUC
        run_frame_model(
            Params(
                epochs=200,
                train_size=600,
                test_size=200,
                hidden_nodes=[176],
                corpus="piano_notes_88_poly_3_to_15_velocity_63_to_127",
                learning_rate=0.005,
                lower=21,
                upper=109,
                padding=0,
                batch_size=4096
            ),
            report_epochs=20,
            pre_p=Params(
                epochs=200,
                train_size=48,
                test_size=2,
                hidden_nodes=[176],
                corpus="piano_notes_88_mono_velocity_95",
                learning_rate=0.03,
                lower=21,
                upper=109,
                padding=0,
                batch_size=4096
            ),
            ui=False
        )
    else:
        assert False


def run_best_rnn(corpus):
    # 0.17156129
    if corpus == "two_piano_one_octave":
        run_sequence_model(
            Params(
                epochs=50,
                train_size=4,
                test_size=1,
                hidden_nodes=[],
                corpus="two_piano_one_octave",
                learning_rate=0.002,
                lower=60,
                upper=72,
                padding=0,
                batch_size=1,
                steps=50,
                hidden=8,
                graph_type="bi_rnn"
            )
        )
    elif corpus == "five_piano_two_middle_octaves":
        # 0.17442912
        run_sequence_model(
            Params(
                epochs=11,
                train_size=150,
                test_size=50,
                hidden_nodes=[],
                corpus="five_piano_two_middle_octaves",
                learning_rate=0.01,
                lower=48,
                upper=72,
                padding=0,
                batch_size=16,
                steps=200,
                hidden=64,
                graph_type="lstm"
            )
        )
    elif corpus == "five_piano_magic":
        # 0.10388491
        run_sequence_model(
            Params(
                epochs=20,
                train_size=400,
                test_size=100,
                hidden_nodes=[],
                corpus="five_piano_magic",
                learning_rate=0.01,
                lower=21,
                upper=109,
                padding=0,
                batch_size=16,
                steps=200,
                hidden=64,
                graph_type="lstm"
            )
        )
    elif corpus == "":
        # 0.15517218
        run_sequence_model(
            Params(
                epochs=95,
                train_size=600,
                test_size=200,
                hidden_nodes=[],
                corpus="piano_notes_88_poly_3_to_15_velocity_63_to_127",
                learning_rate=0.01,
                lower=21,
                upper=109,
                padding=0,
                batch_size=16,
                steps=500,
                hidden=64,
                graph_type="lstm"
            )
        )
    else:
        assert False
