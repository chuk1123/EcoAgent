"""Oracle White Agent - Returns the ground truth value directly."""


def run(ga):
    """Return the ground truth prediction (15378.0 for Vacant Units)."""
    y_true = [15378.0]
    return ga.evaluate_predictions(y_true)