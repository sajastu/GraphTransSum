# stopwords = pkgutil.get_data(__package__, 'smart_common_words.txt')
# stopwords = stopwords.decode('ascii').split('\n')
# stopwords = {key.strip(): 1 for key in stopwords}
import random

import scispacy
import spacy
nlp = spacy.load("en_core_sci_md")


def remove_ack(source, debug=False):
    out = []
    sect_idx = 2
    if debug:
        import pdb;
        pdb.set_trace()

    for sent in source:
        section_txt = sent[sect_idx].lower().replace(':', ' ').replace('.', ' ').replace(';', ' ')
        if \
                'acknowledgment' in section_txt.split() \
                        or 'acknowledgments' in section_txt.split() \
                        or 'acknowledgements' in section_txt.split() \
                        or 'fund' in section_txt.split() \
                        or 'funding' in section_txt.split() \
                        or 'appendices' in section_txt.split() \
                        or 'proof of' in section_txt.split() \
                        or 'related work' in section_txt.split() \
                        or 'previous works' in section_txt.split() \
                        or 'references' in section_txt.split() \
                        or 'figure captions' in section_txt.split() \
                        or 'acknowledgement' in section_txt.split() \
                        or 'appendix' in section_txt.split() \
                        or 'appendix:' in section_txt.split():

            continue

        else:
            out.append(sent)

    return out

def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def get_negative_samples(source_sents, bertScores, sampling_len=10, tokenizer=None):
    def last(n):
        return n[-1]

    def first(n):
        return n[0]

    def sort(tuples):
        return sorted(tuples, key=last)

    def sort_first(tuples):
        return sorted(tuples, key=first)

    def rotate(l, n=1):
        return l[n:] + l[:n]

    out = [(s, b) for s,b in zip(source_sents, bertScores)]

    sorted_sents_bertScore = sort(out)

    sampled_sents = []

    first_half = sorted_sents_bertScore[:int(len(sorted_sents_bertScore) / 2)]
    second_half = sorted_sents_bertScore[int(len(sorted_sents_bertScore) / 2):]
    # source_sents = [([tkn.lower() for tkn in s[1]], s[2], s[0]) for s in sampled_sents]

    ptrs = [0, 0]
    halves = [first_half, second_half]
    while len(sampled_sents) < sampling_len or tokenizer.cal_token_len([([tkn.lower() for tkn in s[1]], s[2], s[0]) for s in sampled_sents]) < 1026:
        try:
            if ptrs[0] > len(halves[0]) - 1:
                del ptrs[0]
                del halves[0]
                continue
        except Exception as e:
            break
        try:
            cand_sent = halves[0][ptrs[0]][:-1]
        except:
            break
        if cand_sent[0][-6] != 1:
            sampled_sents.append(cand_sent[0])
            ptrs[0] += 1

            ptrs = rotate(ptrs)
            halves = rotate(halves)

        else:
            ptrs[0] += 1
            continue

        if tokenizer.cal_token_len([([tkn.lower() for tkn in s[1]], s[2], s[0]) for s in sampled_sents]) > 1026 or len(sampled_sents) == sampling_len:
            break

    sampled_sents = sort_first(sampled_sents)
    # sorted_out = sort_first([s[:-1][0] for s in sorted_sents_bertScore[:sampling_len] if s[:-1][0][-6] != 1])

    # sampling rest 1/2 of sentences from the intermediate.

    # sort the selected sentences based off the sentence index

    return sampled_sents

def get_positive_samples_form_abstracts(paper_id, abstracts):
    return abstracts[paper_id]

def get_negative_samples_from_abstracts(paper_id, abstracts):

    # random.seed(8888)
    abstracts_list = list(abstracts.items())
    random.shuffle(abstracts_list)
    out_abs = abstracts_list[:5]
    sampled_keys = [k for k, v in out_abs]

    if paper_id in sampled_keys:
        out_abs = abstracts_list[:6]
        out_abs = [(k, v) for k,v in out_abs if k != paper_id]
        sampled_keys = [k for k, v in out_abs]

    # for key in sampled_keys:
    #     abstracts.pop(key, None)
    return out_abs

def get_negative_samples_from_references(paper_id, ref_abstracts):
    ref_out = []
    for ref_abs in ref_abstracts[paper_id]:
        ref_tokenized = []
        try:
            if str(ref_abs[0]) == 'b':
                ref_abs = ref_abs.encode('ascii', 'ignore').decode('utf-8')[2:-2]
            else:
                ref_abs = ref_abs
        except:
            ref_abs = ref_abs

        ref_abs_nlp = nlp(ref_abs.lower()).sents

        for sent in ref_abs_nlp:
            ref_sent_tknzed = []
            for tkn in sent:
                ref_sent_tknzed.append(tkn.text)
            ref_tokenized.append(ref_sent_tknzed)


        ref_out.append(ref_tokenized)
    return ref_out


def get_positive_samples(source_sents, tokenizer=None):

    out = []
    for s in source_sents:
        if s[-6] == 1:
           out.append(s)

    return out