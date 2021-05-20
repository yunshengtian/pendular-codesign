from .ilqr import ILQR
from .mppi import MPPI


def get_control(name):
    control_map = {
        'ilqr': ILQR,
        'mppi': MPPI,
    }
    Control = control_map[name]
    return Control
