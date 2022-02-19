import time
from flask import Flask
from pathlib import Path
import json

from drl_tetris.training_state import training_state, cache, is_agent_root

app = Flask(__name__)
state = training_state(me="sidecar", dummy=True)

@app.route('/')
def hello():
    count = cache.incr('sidecar-hits')
    return 'Hello World! I have been seen {} times.\n'.format(count)

@app.route('/keys')
def list_keys():
    ret = ""
    for key in cache.keys():
    # for key in cache.scan_iter():
        ret += f"<br>{key}"
    return ret

@app.route('/stats', defaults={'agent': 'trainer'})
@app.route('/stats/<path:agent>')
def stats(agent):
    #######
    ### MAKE THIS WORK!!!!
    #####
    if is_agent_root(agent):
        state.me = agent
        stats = state.fetch_stats()
        return stats
    return f"invalid scope: {agent}"

@app.route('/redis', defaults={'key': 'trainer'})
@app.route('/redis/<path:key>')
def get_key(key):
    val   = cache.get(key)
    return list_keys() + f"<br>---<br>{key}: {val}"
