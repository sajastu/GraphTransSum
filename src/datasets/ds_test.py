import glob
import json

# ent = json.load(open('/disk1/sajad/datasets/sci/arxiv/inputs/val/cond-mat0309544.json'))
# import pdb;pdb.set_trace()


WR = '/disk1/sajad/datasets/sci/arxivL/splits/reference_network_'


for se in ['train', 'test', 'val']:
    all_refs = {}
    for f in glob.glob(f'/disk1/sajad/datasets/sci/arxivL/splits/reference_network_{se}_*'):
        all_refs.update(json.load(open(f)))
    json.dump(all_refs, open(f'{WR}{se}_all.json', mode='w'))