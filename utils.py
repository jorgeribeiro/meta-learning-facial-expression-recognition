from bson import json_util
import json
import os

from constants import *

def print_json(result):
    print(json.dumps(
        result,
        default=json_util.default, sort_keys=True,
        indent=4, separators=(',', ': ')
    ))

def save_json_result(result, result_name):
    with open(os.path.join(RESULTS_PATH, result_name), 'w') as f:
	    json.dump(
	        result, f,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
	    )

def load_json_result(result_name):
    with open(os.path.join(RESULTS_PATH, result_name), 'r') as f:
        return json.JSONDecoder().decode(
            f.read()
        )