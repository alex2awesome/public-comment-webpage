import numpy as np
from nltk import sent_tokenize
from autometrics.metrics.unieval.scorer import UniEvaluator
from autometrics.metrics.unieval.utils import add_question

import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


def safe_sent_tokenize(text: str):
    """
    Like sent_tokenize, but if it returns an empty list (e.g. for '' or
    no punctuation), we fall back to treating the entire text as one segment.
    """
    sents = sent_tokenize(text)
    return sents if sents else [text]


class SumEvaluator:
    def __init__(self, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for text summarization """
        self.scorer = UniEvaluator(
            model_name_or_path='MingZhong/unieval-sum',
            max_length=max_length,
            device=device,
            cache_dir=cache_dir
        )
        self.task = 'summarization'
        self.dimensions = ['coherence', 'consistency', 'fluency', 'relevance']

    def evaluate(self, data, dims=None, overall=True):
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]
        eval_dims = dims if dims is not None else self.dimensions

        for dim in eval_dims:
            # sentence‐level dims
            if dim in ('consistency', 'fluency'):
                src_list, output_list = [], []
                n_sents = []
                for i in range(n_data):
                    source = data[i]['source'] if dim == 'consistency' else ''
                    # safe tokenize
                    sents = safe_sent_tokenize(data[i]['system_output'])
                    n_sents.append(len(sents))
                    for sent in sents:
                        src_list.append(source)
                        output_list.append(sent)

                inputs = add_question(
                    dimension=dim,
                    output=output_list,
                    src=src_list,
                    task=self.task
                )
                sent_scores = self.scorer.score(inputs)

                # average back to per‐example
                score = []
                idx = 0
                for count in n_sents:
                    chunk = sent_scores[idx: idx + count]
                    score.append(sum(chunk) / count)
                    idx += count

            # summary‐level dims
            elif dim in ('coherence', 'relevance'):
                src_list, output_list, ref_list = [], [], []
                for i in range(n_data):
                    src_list.append(data[i]['source'])
                    output_list.append(data[i]['system_output'])
                    if dim == 'relevance':
                        ref_list.append(data[i]['reference'])
                inputs = add_question(
                    dimension=dim,
                    output=output_list,
                    src=src_list,
                    ref=ref_list,
                    task=self.task
                )
                score = self.scorer.score(inputs)

            else:
                raise NotImplementedError(
                    f"Dimension '{dim}' not implemented for summarization."
                )

            for i in range(n_data):
                eval_scores[i][dim] = score[i]

        if overall:
            for i in range(n_data):
                eval_scores[i]['overall'] = np.mean(list(eval_scores[i].values()))

        return eval_scores


class DialogEvaluator:
    def __init__(self, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for dialogues """
        self.scorer = UniEvaluator(
            model_name_or_path='MingZhong/unieval-dialog',
            max_length=max_length,
            device=device,
            cache_dir=cache_dir
        )
        self.task = 'dialogue'
        self.dimensions = [
            'naturalness', 'coherence',
            'engagingness', 'groundedness', 'understandability'
        ]

    def evaluate(self, data, dims=None, overall=True):
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]
        eval_dims = dims if dims is not None else self.dimensions

        for dim in eval_dims:
            if dim == 'engagingness':
                src_list, output_list, ctx_list = [], [], []
                n_sents = []
                for i in range(n_data):
                    source = data[i]['source']
                    context = data[i]['context']
                    sents = safe_sent_tokenize(data[i]['system_output'])
                    n_sents.append(len(sents))
                    for sent in sents:
                        src_list.append(source)
                        ctx_list.append(context)
                        output_list.append(sent)

                inputs = add_question(
                    dimension=dim,
                    output=output_list,
                    src=src_list,
                    context=ctx_list,
                    task=self.task
                )
                sent_scores = self.scorer.score(inputs)

                score = []
                idx = 0
                for count in n_sents:
                    chunk = sent_scores[idx: idx + count]
                    # here we sum (per original code)
                    score.append(sum(chunk))
                    idx += count

            elif dim in ['naturalness', 'coherence', 'groundedness', 'understandability']:
                src_list, output_list, ctx_list = [], [], []
                for i in range(n_data):
                    src_list.append(data[i]['source'] if dim == 'coherence' else '')
                    output_list.append(data[i]['system_output'])
                    ctx_list.append(data[i]['context'] if dim == 'groundedness' else '')

                inputs = add_question(
                    dimension=dim,
                    output=output_list,
                    src=src_list,
                    context=ctx_list,
                    task=self.task
                )
                score = self.scorer.score(inputs)
            else:
                raise NotImplementedError(
                    f"Dimension '{dim}' not implemented for dialogue."
                )

            for i in range(n_data):
                eval_scores[i][dim] = score[i]

        if overall:
            for i in range(n_data):
                eval_scores[i]['overall'] = np.mean(list(eval_scores[i].values()))

        return eval_scores


class D2tEvaluator:
    def __init__(self, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for data-to-text """
        self.scorer = UniEvaluator(
            model_name_or_path='MingZhong/unieval-sum',
            max_length=max_length,
            device=device,
            cache_dir=cache_dir
        )
        self.task = 'data2text'
        self.dimensions = ['naturalness', 'informativeness']

    def evaluate(self, data, dims=None, overall=True):
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]
        eval_dims = dims if dims is not None else self.dimensions

        for dim in eval_dims:
            output_list = [d['system_output'] for d in data]
            ref_list    = [d['reference']     for d in data]
            inputs = add_question(
                dimension=dim,
                output=output_list,
                ref=ref_list,
                task=self.task
            )
            score = self.scorer.score(inputs)

            for i in range(n_data):
                eval_scores[i][dim] = score[i]

        if overall:
            for i in range(n_data):
                eval_scores[i]['overall'] = np.mean(list(eval_scores[i].values()))

        return eval_scores


class FactEvaluator:
    def __init__(self, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for factual consistency detection """
        self.scorer = UniEvaluator(
            model_name_or_path='MingZhong/unieval-fact',
            max_length=max_length,
            device=device,
            cache_dir=cache_dir
        )
        self.task = 'fact'
        self.dim = 'consistency'

    def evaluate(self, data):
        """
        Get the factual consistency score (only 1 dimension for this task).
        """
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]

        # 1) Flatten every sentence, while building a matching src_expanded.
        output_list = []
        src_expanded = []
        n_sents = []
        for i, sample in enumerate(data):
            source = sample['source']
            sents = safe_sent_tokenize(sample['system_output'])
            n_sents.append(len(sents))
            for sent in sents:
                output_list.append(sent)
                src_expanded.append(source)

        # sanity check
        assert len(output_list) == len(src_expanded), (
            f"output_list ({len(output_list)}) != src_expanded ({len(src_expanded)})"
        )

        # 2) Build inputs and score every sentence
        inputs = add_question(
            dimension=self.dim,
            output=output_list,
            src=src_expanded,
            task=self.task
        )
        sent_scores = self.scorer.score(inputs)

        # 3) Re-aggregate per-example averages
        scores = []
        idx = 0
        for count in n_sents:
            # safe_sent_tokenize ensures count >= 1
            chunk = sent_scores[idx: idx + count]
            scores.append(sum(chunk) / count)
            idx += count

        # 4) Fill in the returns
        for i, sc in enumerate(scores):
            eval_scores[i][self.dim] = sc

        return eval_scores


def get_evaluator(task, max_length=1024, device='cuda:0', cache_dir=None):
    assert task in ['summarization', 'dialogue', 'data2text', 'fact']
    if task == 'summarization':
        return SumEvaluator(max_length, device, cache_dir)
    if task == 'dialogue':
        return DialogEvaluator(max_length, device, cache_dir)
    if task == 'data2text':
        return D2tEvaluator(max_length, device, cache_dir)
    if task == 'fact':
        return FactEvaluator(max_length, device, cache_dir)