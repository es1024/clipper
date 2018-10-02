import json
import numpy as np

EXP3_LEARNING_RATE = 0.05

def draw(w):
    sum_w = sum(w.values())
    v = np.random.uniform(0.0, sum_w)
    for k in w.keys():
        if v <= w[k]:
            return k
        v -= w[k]

def model_to_str(m):
    return '{}-{}'.format(m['name'], m['id'])

def str_to_model(s):
    name, model_id = s.rsplit('-', 1)
    return {'name': name, 'id': model_id}

def load_state(state, query):
    if state == 'state':
        state = '{}'
    w = json.loads(state)
    candidate_models = set(model_to_str(m) for m in query['candidate_models'])
    for k in w.keys():
        if k not in candidate_models:
            del w[k]
    avg_w = sum(w.values()) / len(w) if w else 1.0
    for k in candidate_models:
        if k not in w:
            w[k] = avg_w
    sum_w = sum(w.values())
    for k in w.keys():
        w[k] = w[k] / sum_w
    return w

def select(state, query):
    if state == 'state':
        r = [query['candidate_models'][np.random.randint(0, len(query['candidate_models']))]]
        return [query['candidate_models'][np.random.randint(0, len(query['candidate_models']))]]
    w = load_state(state, query)
    return [str_to_model(draw(w))]

def update(state, query):
    w = load_state(state, query)
    m = model_to_str(query['selected_models'][0])
    loss = 0.0 if np.abs(query['model_outputs'][0][0] - query['feedback']['label']) < 0.5 else 1.0
    w[m] = w[m] * np.exp(-EXP3_LEARNING_RATE * loss / w[m])
    sum_w = sum(w.values())
    for k in w.keys():
        w[k] = w[k] / sum_w
    return json.dumps(w)


