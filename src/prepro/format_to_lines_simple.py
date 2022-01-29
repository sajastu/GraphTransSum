import glob
import json
from multiprocessing import Pool

from tqdm import tqdm

from prepro.FIXED_KEYS import *
from prepro.data_builder import check_path_existence


def format_to_lines_simple(args):
    if args.dataset != '':
        corpuses_type = [args.dataset]
    else:
        corpuses_type = ['train', 'val', 'test']

    for corpus_type in corpuses_type:
        files = []
        for f in glob.glob(args.raw_path +'/*.json'):
            files.append(f)

        corpora = {corpus_type: files}
        for corpus_type in corpora.keys():
            a_lst = [(f, args.keep_sect_num) for f in corpora[corpus_type]]
            pool = Pool(args.n_cpus)
            dataset = []
            p_ct = 0
            all_papers_count = 0
            curr_paper_count = 0
            check_path_existence(args.save_path)

            # for a in a_lst:
            #     _format_to_lines(a)

            for d in tqdm(pool.imap_unordered(_format_to_lines, a_lst), total=len(a_lst)):
                # d_1 = d[1]
                if d is not None:
                    all_papers_count+=1
                    curr_paper_count+=1

                    # dataset.extend(d[0])
                    dataset.append(d)
                    # import pdb;pdb.set_trace()
                    # if (len(dataset) > args.shard_size):
                    if (curr_paper_count > args.shard_size):
                        pt_file = "{:s}{:s}.{:d}.json".format(args.save_path + '', corpus_type, p_ct)
                        print(pt_file)
                        with open(pt_file, 'w') as save:
                            save.write(json.dumps(dataset))
                            print('data len: {}'.format(len(dataset)))
                            p_ct += 1
                            dataset = []
                        curr_paper_count = 0


            pool.close()
            pool.join()

            if (len(dataset) > 0):
                pt_file = "{:s}{:s}.{:d}.json".format(args.save_path + '', corpus_type, p_ct)
                print(pt_file)
                # all_papers_count += len(dataset)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1

                    dataset = []
            print('Processed {} papers for {} set'.format(all_papers_count, corpus_type))





def _format_to_lines(params):
    src_path, keep_sect_num = params

    def load_json(src_json):
        # print(src_json)
        ent = json.load(open(src_json))
        id = ent['id']
        sentences = [s['sent_text'] for s in ent[SRC_SENTS_KEY]]
        labels = [s['is_oracle'] for s in ent[SRC_SENTS_KEY]]
        gold = ent[GOLD_KEY]
        return id, sentences, labels, gold

    id, src_sents, labels, tgt = load_json(src_path)

    return {ID_KEY: id, SRC_SENTS_KEY: src_sents, GOLD_KEY: tgt, SRC_SENTS_LABELS: labels}