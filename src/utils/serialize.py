from typing import Dict
import json

import numpy as np


class npEncoder(json.JSONEncoder):
    def default(self, obj: np.int32) -> int:
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def load_json(file: str) -> Dict:
    """Load json file.
    :args:
         - file - str path to file
    :returns:
         -"""

    with open(file, "r") as f:
        return json.load(f)
