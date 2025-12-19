import re
import sys
import string
import numpy as np
import collections
import pandas as pd

from tqdm import tqdm
from task import Task
from glue_utils import Glue_utils
from xglue_utils import Xglue_utils
from partition import get_partition_map
from scipy.stats import pearsonr, spearmanr
from seqeval.metrics import precision_score
from sklearn.datasets import fetch_20newsgroups
from datasets import Dataset, DatasetDict, load_dataset
from seqeval.metrics import f1_score as seqeval_f1_score
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import matthews_corrcoef, ndcg_score


def get_fed_data_info(args, tokenizer):
    guids = None
    label_names = []
    validation_dataset = None
    if args.dataset == "20_newsgroups":
        tokenized_dataset, label_list = get_20_newsgroups_tokenizer_dataset(tokenizer=tokenizer,
                                                                            max_length=args.max_length)
        num_labels = 20
        partition_map = get_partition_map(partition=args.partition, label=args.label,
                                          tokenized_datasets=tokenized_dataset,
                                          num_client=args.num_client, alpha=args.alpha,
                                          label_list=label_list, num_labels=num_labels)
        num_labels = [num_labels]
        validation_key = "test"
        metric = ["accuracy"]
        task = Task.SequenceClassification

    elif args.dataset == "glue":
        glue = Glue_utils(args.subdataset)
        tokenized_dataset = glue.get_tokenizer_dataset(tokenizer=tokenizer, max_length=args.max_length)
        num_labels = glue.num_labels
        partition_map = get_partition_map(partition=args.partition, label=args.label,
                                          tokenized_datasets=tokenized_dataset,
                                          num_client=args.num_client, alpha=args.alpha,
                                          label_list=glue.get_label_list(), num_labels=num_labels)
        num_labels = [num_labels]
        metric = glue.get_metric()
        validation_key = glue.validation_key
        task = Task.SequenceClassification

    elif args.dataset == "xglue":
        xglue = Xglue_utils(args.subdataset)
        tokenized_dataset = xglue.get_tokenizer_dataset(tokenizer=tokenizer, max_length=args.max_length)
        num_labels = xglue.get_label_number()
        if not args.partition or not args.label:
            partition_map = get_partition_map(partition=args.partition, label=args.label,
                                              tokenized_datasets=tokenized_dataset,
                                              num_client=args.num_client, alpha=args.alpha,
                                              label_list=[], num_labels=num_labels)
            metric = xglue.get_metric()
            validation_key = 'validation'
            task = xglue.task_type
            label_names = xglue.label_names
            validation_dataset = xglue.get_raw_dataset_validation()
            guids = xglue.get_wpr_grid()
        else:
            raise KeyError("xglue does not support split by label")
    elif args.dataset == "squad":
        tokenized_dataset, validation_dataset = get_squad_tokenizer_dataset(tokenizer=tokenizer)
        num_labels = 0
        validation_key = 'validation'
        metric = ["exact_match", "f1"]
        task = Task.QuestionAnswering
        partition_map = get_partition_map(partition=args.partition, label=args.label,
                                          tokenized_datasets=tokenized_dataset,
                                          num_client=args.num_client, alpha=args.alpha,
                                          label_list=[], num_labels=num_labels)
    else:
        exit('No dataset {} erro in get_fed_data_info'.format(args.dataset))
    return tokenized_dataset, partition_map, num_labels, metric, validation_key, task, label_names, \
        validation_dataset, guids


def get_20_newsgroups_tokenizer_dataset(tokenizer, max_length):
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    train_df = pd.DataFrame({'text': newsgroups_train['data'], 'label': newsgroups_train['target']})
    test_df = pd.DataFrame({'text': newsgroups_test['data'], 'label': newsgroups_test['target']})

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    full_dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

    tokenized_dataset = full_dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['text'])
    tokenized_dataset.set_format("torch")
    return tokenized_dataset, newsgroups_train.target


def get_squad_tokenizer_dataset(tokenizer, max_length=256, stride=128):
    raw_datasets = load_dataset(path='./data/squad')

    def preprocess_training_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs

    train_dataset = raw_datasets["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    validation_dataset = raw_datasets["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )
    tokenized_datasets = DatasetDict({"train": train_dataset, "validation": validation_dataset})
    return tokenized_datasets, raw_datasets["validation"]


def get_spearman(preds, labels):
    preds = preds[:, 0]
    spearman_corr = float(spearmanr(preds, labels)[0])
    return {"spearmanr": spearman_corr}


def get_pearson(preds, labels):
    preds = preds[:, 0]
    pearson_corr = float(pearsonr(preds, labels)[0])
    return {"pearson": pearson_corr}


def get_matthews_correlation(preds, labels):
    preds = np.argmax(preds, axis=-1)
    return {"matthews_correlation": matthews_corrcoef(labels, preds)}


def get_accuracy(preds, labels):
    preds = np.argmax(preds, axis=-1)
    acc = float((preds == labels).mean())
    return {"acc": acc}


def get_f1(preds, labels):
    preds = np.argmax(preds, axis=-1)
    f1 = float(sklearn_f1_score(y_true=labels, y_pred=preds))
    return {"f1": f1}


def get_f1_m(preds, labels, label_names):
    preds = np.argmax(preds, axis=-1)
    true_labels = [[label_names[la] for la in lab if la != -100] for lab in labels]
    true_predictions = [
        [label_names[p] for (p, la) in zip(prediction, lab) if la != -100]
        for prediction, lab in zip(preds, labels)
    ]
    f1 = float(seqeval_f1_score(y_true=true_labels, y_pred=true_predictions))
    return {"f1": f1}


def get_precision(preds, labels, label_names):
    preds = np.argmax(preds, axis=-1)
    true_labels = [[label_names[la] for la in lab if la != -100] for lab in labels]
    true_predictions = [
        [label_names[p] for (p, la) in zip(prediction, lab) if la != -100]
        for prediction, lab in zip(preds, labels)
    ]
    precision = precision_score(y_true=true_labels, y_pred=true_predictions)
    return {"precision": precision}


def format_prediction(predictions, references):
    pred_dict = {prediction["id"]: prediction["prediction_text"] for prediction in predictions}
    dataset = [
        {
            "paragraphs": [
                {
                    "qas": [
                        {
                            "answers": [{"text": answer_text} for answer_text in ref["answers"]["text"]],
                            "id": ref["id"],
                        }
                        for ref in references
                    ]
                }
            ]
        }
    ]
    return pred_dict, dataset


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def compute_score(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                if qa["id"] not in predictions:
                    message = "Unanswered question " + qa["id"] + " will receive score 0."
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x["text"], qa["answers"]))
                prediction = predictions[qa["id"]]
                exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = exact_match / total
    f1 = f1 / total

    return {"eval_exact_match": exact_match, "eval_f1": f1}


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_ndcg(preds, labels, guids):
    preds = np.argmax(preds, axis=-1)
    ndcgs = []
    query2content = {}
    for guid, pred, label in zip(guids, preds, labels):
        query = guid
        if query not in query2content:
            query2content[query] = [[float(pred)], [float(label)]]
        else:
            query2content[query][0].append(float(pred))
            query2content[query][1].append(float(label))

    for key in query2content.keys():
        if len(query2content[key][1]) < 2 or len(query2content[key][0]) < 2:
            continue
        ndcgs.append(ndcg_score(np.asarray([query2content[key][1]]), np.asarray([query2content[key][0]])))
    return {"ndcg": np.array(ndcgs).mean()}


class Metrics_trainer(object):
    def __init__(self, metric_name, label_names, grids):
        self.metric_name = metric_name
        self.label_names = label_names
        self.grids = grids

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        evaluate_result = {}
        for metric in self.metric_name:
            if metric == "matthews_correlation":
                evaluate_result.update(get_matthews_correlation(predictions, labels))
            elif metric == "pearson":
                evaluate_result.update(get_pearson(predictions, labels))
            elif metric == "spearman":
                evaluate_result.update(get_spearman(predictions, labels))
            elif metric == "accuracy":
                evaluate_result.update(get_accuracy(predictions, labels))
            elif metric == "f1":
                evaluate_result.update(get_f1(predictions, labels))
            elif metric == "f1_m":
                evaluate_result.update(get_f1_m(predictions, labels, self.label_names))
            elif metric == "precision":
                evaluate_result.update(get_precision(predictions, labels, self.label_names))
            elif metric == "nDCG":
                evaluate_result.update(get_ndcg(predictions, labels, self.grids))
            else:
                exit('No metric {} erro in compute_metrics'.format(metric))
        return evaluate_result

    def compute_metrics_predictions(self, predictions, data, result: dict, validation_dataset):
        start_logits, end_logits = predictions
        validation = data['validation']
        n_best = 20
        max_answer_length = 30
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(validation):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(validation_dataset):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = validation[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
                end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                                end_index < start_index
                                or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0]: offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in validation_dataset]

        if 'f1' in self.metric_name and 'exact_match' in self.metric_name:
            prediction, label = format_prediction(predicted_answers, theoretical_answers)
            evaluate_result = compute_score(predictions=prediction, dataset=label)
        else:
            exit('No metric_name {} erro in compute_metrics_predictions'.format(self.metric_name))
        print(evaluate_result)
        result.update(evaluate_result)
        return evaluate_result
