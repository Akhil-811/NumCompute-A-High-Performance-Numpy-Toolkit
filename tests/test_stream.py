import numpy as np

from numcompute.tree import (
    DecisionTreeClassifier
)

from numcompute.stream import (
    StreamTrainer
)


def test_split_stream():

    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)

    chunks = list(
        StreamTrainer.split_stream(
            X,
            y,
            chunk_size=3
        )
    )

    assert len(chunks) == 4


def test_stream_fit():

    X = np.random.rand(20, 2)

    y = np.random.randint(
        0,
        2,
        20
    )

    model = DecisionTreeClassifier()

    trainer = StreamTrainer(model)

    trainer.fit_stream(
        X,
        y,
        chunk_size=5
    )

    logs = trainer.get_logs()

    assert len(logs["accuracy"]) > 0


def test_stream_summary():

    X = np.random.rand(20, 2)

    y = np.random.randint(
        0,
        2,
        20
    )

    model = DecisionTreeClassifier()

    trainer = StreamTrainer(model)

    trainer.fit_stream(
        X,
        y,
        chunk_size=5
    )

    summary = trainer.summary()

    assert "final_accuracy" in summary


def test_stream_evaluate():

    X = np.random.rand(20, 2)

    y = np.random.randint(
        0,
        2,
        20
    )

    model = DecisionTreeClassifier()

    trainer = StreamTrainer(model)

    trainer.fit_stream(
        X,
        y
    )

    results = trainer.evaluate(
        X,
        y
    )

    assert "accuracy" in results


def test_reset_logs():

    model = DecisionTreeClassifier()

    trainer = StreamTrainer(model)

    trainer.logs["accuracy"] = [1]

    trainer.reset_logs()

    assert trainer.logs["accuracy"] == []