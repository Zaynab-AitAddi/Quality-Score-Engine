from quality_engine.calibration import fit_calibration, LinearCalibrator


def test_fit_calibration_linear():
    raw = [10, 20, 30, 40, 50]
    human = [15, 25, 35, 45, 55]  # human = raw + 5
    calib = fit_calibration(raw, human)
    assert isinstance(calib, LinearCalibrator)
    # slope approximately 1 and intercept ~5
    assert round(calib.slope, 3) == 1.0
    assert round(calib.intercept, 3) == 5.0
