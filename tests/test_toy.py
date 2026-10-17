from project.toy import LinearModel, Params, load_checkpoint, mse, sample_points, save_checkpoint, sgd_step


def test_sgd_on_the_toy_data_approaches_the_true_line():
    points = sample_points(seed=0, count=256, noise=0.0)
    params = Params(LinearModel().init_weight, LinearModel().init_bias)
    start = mse(params, points)
    for step in range(2000):
        x, y = points[step % len(points)]
        params = sgd_step(params, x, y, lr=0.05)
    assert mse(params, points) < start / 100


def test_a_checkpoint_round_trips(tmp_path):
    save_checkpoint(str(tmp_path / "step-5"), Params(weight=2.5, bias=-1.0))
    assert load_checkpoint(str(tmp_path / "step-5")) == Params(weight=2.5, bias=-1.0)
