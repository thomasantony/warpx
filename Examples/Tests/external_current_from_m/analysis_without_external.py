#!/usr/bin/env python3

from pathlib import Path

import numpy as np


def value(name):
    data = np.atleast_2d(np.loadtxt(Path("diags/reducedfiles") / f"{name}.txt"))
    return float(data[-1, 2])


external = value("ExternalCurrentL2")
plasma = value("PlasmaCurrentL2")
total = value("TotalCurrentL2")
assert external == 0.0
assert total > 0.0
assert np.isclose(plasma, total, rtol=1.0e-14, atol=0.0)
assert np.isclose(total - plasma - external, 0.0, rtol=0.0, atol=1.0e-30)
