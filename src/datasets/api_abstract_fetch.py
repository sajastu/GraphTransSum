import glob
import json
import os
import pickle
import sys
import time
from urllib.request import Request, urlopen
from multiprocessing import Pool
from langdetect import detect
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
# import requests
from joblib import Parallel, delayed
from multiprocessing.pool import ThreadPool
from threading import BoundedSemaphore as BoundedSemaphore, Timer

BASE_DS_DIR = "/disk1/sajad/datasets/sci/arxivL/splits/"


class RatedSemaphore(BoundedSemaphore):
    """Limit to 1 request per `period / value` seconds (over long run)."""
    def __init__(self, value=1, period=1):
        BoundedSemaphore.__init__(self, value)
        t = Timer(period, self._add_token_loop,
                  kwargs=dict(time_delta=float(period) / value))
        t.daemon = True
        t.start()

    def _add_token_loop(self, time_delta):
        """Add token every time_delta seconds."""
        while True:
            try:
                BoundedSemaphore.release(self)
            except ValueError: # ignore if already max possible value
                pass
            time.sleep(time_delta) # ignore EINTR

    def release(self):
        pass # do nothing (only time-based release() is allowed)

def _scholar_endpoint(scholar_id):
    return f"https://api.semanticscholar.org/v1/paper/{scholar_id}"

def _arxiv_endpoint(arxiv_id):
    # return f"https://api.semanticscholar.org/v1/paper/arXiv:{arxiv_id}"
    return f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}/references?fields=title,abstract"


def _get_paper_info(scholar_id):
    req = Request(_scholar_endpoint(scholar_id), headers={'User-Agent': 'Mozilla/5.0'})
    r = urlopen(req).read()
    j_response = json.loads(r)
    return j_response


def _get_references(endpoint):

    try:
        req = Request(endpoint, headers={'User-Agent': 'Chrome/5.0'})
        res = urlopen(req).read()
        j_response = json.loads(res)
        if 'references' in j_response.keys():
            references = j_response['references']
            abstract_list = []
            for reference in references:
                ref_id = reference['paperId']
                ref_arxiv_id = reference['arxivId']
                abstract_list.append(
                    {
                        'scholar_id': ref_id,
                        'arxiv_id': ref_arxiv_id
                    }
                )
        return {
            'paper_id': endpoint.replace('https://api.semanticscholar.org/v1/paper/arXiv:',''),
            'references': abstract_list
        }, None

    except Exception as e:
        return None, e

def _get_references_with_abs(endpoint):

    try:
        req = Request(endpoint, headers={'User-Agent': 'Chrome/5.0'})
        res = urlopen(req).read()
        j_response = json.loads(res)
        if 'data' in j_response.keys():

            return {
                'paper_id': endpoint.replace('https://api.semanticscholar.org/graph/v1/paper/arXiv:', '').replace('/references?fields=title,abstract', ''),
                'references': j_response['data']
            }, None

    except Exception as e:
        return None, e


def check_id(arxiv_id):
    arxiv_id = arxiv_id.strip()
    # check to see if the first char is alphabet

    alphabet = ''
    number = ''

    if not arxiv_id[0].isnumeric():
        position = 0

        while not arxiv_id[position].isnumeric():
            alphabet += arxiv_id[position]
            position +=1

        # the rest is number
        number = arxiv_id[position:]
        return alphabet + '/' + number

    else:
        return arxiv_id





def call_api(abstract_ids, wr_file_dir, paper_references=None, save_file=True, return_ref=False):

    error_wait = False

    if not os.path.exists(wr_file_dir):
        abstract_list = {}
    else:
        abstract_list = paper_references.copy()
    counter = 0
    for j, old_abs in tqdm(enumerate(abstract_ids), total=len(abstract_ids)):
        abs = check_id(old_abs)
        out, error = _get_references_with_abs(_arxiv_endpoint(abs))
        if error is None:
            error_wait = False
            abstract_list[old_abs] = out['references']
            counter += 1
        else:
            if 'Not Found' not in error.reason:
                error_wait = True

        if error_wait:
            print('Error emitted... waiting...')
            if counter > 0 and save_file:
                print(f'Saved papers {(counter)}')
                json.dump(abstract_list, open(wr_file_dir, mode='w'))
                counter = 0
            time.sleep(6 * 60)


        # if j >= 95 and j % 95 == 0:
        #     if counter > 0:
        #         print(f'Saved papers {(counter)}')
        #         json.dump(abstract_list, open(wr_file_dir, mode='w'))
        #         counter = 0
        #     print('Reached max limit... stopping for 5.5 minutes')
        #
        #     time.sleep(6 * 60)


        if j== len(abstract_ids) - 1 and save_file:
            if counter > 0:
                print(f'Saved papers {(counter)}')
                json.dump(abstract_list, open(wr_file_dir, mode='w'))
                counter = 0
