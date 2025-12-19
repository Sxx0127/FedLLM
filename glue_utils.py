from datasets import load_dataset, DatasetDict


class Glue_utils(object):
    def __init__(self, subdataset):
        super().__init__()
        self.task_to_keys = {
            "cola": ("sentence", None),
            "mnli": ("premise", "hypothesis"),
            "mrpc": ("sentence1", "sentence2"),
            "qnli": ("question", "sentence"),
            "qqp": ("question1", "question2"),
            "rte": ("sentence1", "sentence2"),
            "sst2": ("sentence", None),
            "stsb": ("sentence1", "sentence2"),
            "wnli": ("sentence1", "sentence2"),
        }
        self.subdataset = subdataset
        validation_keys = {
            "mnli_mismatched": "validation_mismatched",
            "mnli": "validation_matched",
            "mnli_matched": "validation_matched"
        }
        self.validation_key = validation_keys.get(subdataset, "validation")
        if self.subdataset == "mnli_matched":
            self.subdataset = "mnli"
        raw_dataset = load_dataset("./data/glue", subdataset, cache_dir='./data/glue')
        self.raw_dataset = DatasetDict({'train': raw_dataset['train'], 'validation': raw_dataset[self.validation_key]})
        self.validation_key = 'validation'
        self.sentence1_key, self.sentence2_key = self.task_to_keys[subdataset]
        self.num_labels = 3 if subdataset.startswith("mnli") else 1 if subdataset == "stsb" else 2
        self.metric_name = []

    def get_tokenizer_dataset(self, tokenizer, max_length: int):
        def preprocess_function(examples):
            if self.sentence2_key is None:
                return tokenizer(examples[self.sentence1_key], padding='max_length', truncation=True,
                                 return_tensors="pt", max_length=max_length)
            return tokenizer(examples[self.sentence1_key], examples[self.sentence2_key], padding='max_length',
                             truncation=True, return_tensors="pt", max_length=max_length)

        tokenized_datasets = self.raw_dataset.map(preprocess_function, batched=True)
        return tokenized_datasets

    def get_label_list(self):
        return self.raw_dataset["train"]['label']

    def get_metric(self):
        if self.subdataset == "mnli_mismatched" or self.subdataset == "mnli_matched":
            self.subdataset = "mnli"
        if self.subdataset == "cola":
            self.metric_name = ["accuracy"]
        elif self.subdataset == "stsb":
            self.metric_name = ["pearson", "spearman"]
        elif self.subdataset in ["mrpc", "qqp"]:
            self.metric_name = ["accuracy", "f1"]
        elif self.subdataset in ["sst2", "mnli", "mnli_mismatched", "mnli_matched", "qnli", "rte", "wnli"]:
            self.metric_name = ["accuracy"]
        else:
            raise KeyError(
                "You should supply a configuration name selected in "
                '["sst2", "mnli", "mnli_mismatched", "mnli_matched",'
                '"cola", "stsb", "mrpc", "qqp", "qnli", "rte", "wnli"]'
            )
        return self.metric_name
