
import json
import numpy as np

class JSONNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JSONNumpyEncoder, self).default(obj)
        
def json_dumps(obj, **params):
    return json.dumps(obj, cls=JSONNumpyEncoder, **params)

if __name__ == '__main__':
    pass