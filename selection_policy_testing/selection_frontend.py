import threading
from collections import deque
import json
import sys
import redis
import zmq
import datetime
import numpy as np

def default_select(state, query):
    return query['candidate_models']

def default_combine(state, query):
    if len(query['model_outputs']) > 0:
        return query['model_outputs']
    else:
        return "Default Output"

def ensemble_combine(state, query):
    if len(query['model_outputs']) > 0:
        return np.average(np.array(query['model_outputs']))
    else:
        return "Default Output"

def ab_select(state, query):
    return [query['candidate_models'][np.random.randint(0, len(query['candidate_models']))]]

def default_feedback_select(state, query):
    # with open('/tmp/foo.log', 'a') as f: f.write(repr(query) + '\n')
    return query['candidate_models']

def default_update(state, query):
    # with open('/tmp/foo.log', 'a') as f: f.write(repr(state) + '\n')
    return state

# Have combine return a single string
# Check if preds and return default output if not

policies = {'default':{'select':default_select, 'combine':default_combine, 'feedback-select': default_feedback_select, 'update': default_update},
        'ensemble':{'select':default_select, 'combine':ensemble_combine, 'feedback-select': default_feedback_select, 'update': default_update},
        'ab':{'select':ab_select, 'combine':default_combine, 'feedback-select': default_feedback_select, 'update': default_update}}

class Cache:
    def __init__(self, refcounts=False):
        self.cache = {}
        self.rCount = refcounts

    def __getitem__(self, item):
        if item in self.cache:
            return self.cache[item]

    def __setitem__(self, key, value):
        if key not in self.cache or not self.rCount:
            self.cache[key] = value
        else:
            self.cache[key] = (self.cache[key][0], self.cache[key][1] + 1)

    def pop(self, key):
        if key not in self.cache:
            return True
        state = self.cache[key]
        if self.rCount:
            self.cache[key] = (self.cache[key][0], self.cache[key][1] - 1)
            # To keep the dictionary as small as possible.
            if self.cache[key][1] == 0:
                del self.cache[key]
        else:
            del self.cache[key]
        return state

    def popstate(self, key):
        return self.pop(key)[0]

class Reciever (threading.Thread):
    def __init__(self, select_q, combine_q, sockt):
        super(Reciever, self).__init__()
        self.sq = select_q
        self.cq = combine_q
        self.sock = sockt
        self.sock.connect('tcp://localhost:8080')

    def run(self):
        while True:
            query = self.sock.recv_json()
            if query['msg'] == 'select':
                self.sq.append(query)
                print('append query', query['query_id'])
            elif query['msg'] == 'combine':
                self.cq.append(query)
                print('append cquery', query['query_id'])
            elif query['msg'] == 'feedback-select':
                self.sq.append(query)
                print('append feedback query', query['query_id'])
            elif query['msg'] == 'update':
                self.cq.append(query)
                print('append feedback update', query['query_id'])

class Sender (threading.Thread):
    def __init__(self, send_que, sock):
        super(Sender, self).__init__()
        self.sq = send_que
        self.sock = sock
        self.sock.connect('tcp://localhost:8083')

    def run(self):
        while True:
            if len(self.sq) > 0:
                query = self.sq.popleft()
                if query['msg'] == 'exec':
                    msg = [str(query['query_id'])]
                    for model in query['mids']:
                        msg += [model['name'].encode('utf-8'), model['id'].encode('utf-8')]
                    self.sock.send_multipart(msg)
                else:
                    self.sock.send_json(query)

class SelectionPolicy(threading.Thread):
    def __init__(self, query_queue, send_queue, redis_inst, query_cache, id_cache):
        super(SelectionPolicy, self).__init__()
        self.query_queue = query_queue
        self.redis_inst = redis_inst
        self.send_queue = send_queue
        self.query_cache = query_cache
        self.id_cache = id_cache

    def run(self):
        while True:
            if len(self.query_queue) > 0:
                query = self.query_queue.popleft()
                (timestamp, state) = eval(self.redis_inst.lindex(query['user_id'], 0))
                self.query_cache[(query['user_id'], timestamp)] = (state, 1)
                self.id_cache[query['query_id']] = (query['user_id'], timestamp)
                try:
                    select = policies[query['selection_policy']][query['msg']]
                except KeyError:
                    select = policies['default']['select']
                new_query = {'query_id': query['query_id'], 'msg': 'exec', 'mids': select(state, query)}
                self.send_queue.append(new_query)

class Combiner (threading.Thread):
    def __init__(self, query_queue, send_queue, redis_inst, query_cache, id_cache):
        super(Combiner, self).__init__()
        self.query_queue = query_queue
        self.send_queue = send_queue
        self.redis_inst = redis_inst
        self.query_cache = query_cache
        self.id_cache = id_cache

    def run(self):
        while True:
            if len(self.query_queue) > 0:
                query = self.query_queue.popleft()
                if query['msg'] == 'combine':
                    ret_key = 'final_prediction'
                else:
                    ret_key = 'new_state'
                user_id, timestamp = self.id_cache[query['query_id']]
                state = self.query_cache[(user_id, timestamp)][0]
                err = None
                try:
                    func = policies[query['selection_policy']][query['msg']]
                except KeyError:
                    err = "Selection Policy not found. Default used.'"
                    func = policies['default'][query['msg']]
                res = func(state, query)
                if query['msg'] == 'combine':
                    new_query = {'msg':'return', 'final_prediction': res}
                    if err != None:
                        new_query['combine_error'] = err
                    self.send_queue.append(new_query)
                elif query['msg'] == 'update':
                    self.redis_inst.lpush(user_id, (datetime.datetime.now(), res))

                self.query_cache.pop(self.id_cache[query['query_id']])
                self.id_cache.pop(query['query_id'])

if __name__ == '__main__':
    a = sys.argv
    re = redis.Redis(host=a[1], port=int(a[2]))
    re.lpush(0, (datetime.datetime.now(), b'state'))
    combine = policies['default']['combine']
    select = policies['default']['select']
    select_queue = deque()
    combine_queue = deque()
    send_queue = deque()
    ctx = zmq.Context()
    recieve_sock = ctx.socket(zmq.PAIR)
    send_sock = ctx.socket(zmq.PAIR)
    query_cache = Cache(refcounts=True)
    id_cache = Cache()
    reciever = Reciever(select_queue, combine_queue, recieve_sock)
    sender = Sender(send_queue, send_sock)
    sel_pol = SelectionPolicy(select_queue, send_queue, re, query_cache, id_cache)
    combiner = Combiner(combine_queue, send_queue, re, query_cache, id_cache)
    reciever.start()
    sel_pol.start()
    combiner.start()
    sender.start()
