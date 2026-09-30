"""Test aesthetic calibration fitting and handling of invalid calibrator files."""

from pathlib import Path

from quality_engine.calibration import LinearCalibrator, fit_calibration, load_calibrator


def test_fit_calibration_linear():
    raw = [10, 20, 30, 40, 50]
    human = [15, 25, 35, 45, 55]  # human = raw + 5
    calib = fit_calibration(raw, human)
    assert isinstance(calib, LinearCalibrator)
    # slope approximately 1 and intercept ~5
    assert round(calib.slope, 3) == 1.0
    assert round(calib.intercept, 3) == 5.0


def test_load_calibrator_ignores_invalid_file(tmp_path: Path):
    calibration_file = tmp_path / "calibrator.yaml"
    calibration_file.write_text("invalid: [yaml", encoding="utf-8")

    assert load_calibrator(calibration_file) is None
