"""Course-weighted mean of the student's final grades from strictly earlier terms."""

import numpy as np


def previous_average(frame, student="student", semester="semester", final="final"):
    """Calculate each row's prior mean, keeping every row of the current term out."""
    terms = frame.groupby([student, semester], sort=True)[final].agg(["sum", "count"])
    prior = terms.groupby(level=student).cumsum() - terms
    mean = prior["sum"].div(prior["count"].replace(0, np.nan)).rename("prior_average")
    return frame.join(mean, on=[student, semester])["prior_average"]
