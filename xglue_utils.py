from task import Task
from xglue import _LANGUAGES
from datasets import load_dataset, concatenate_datasets, DatasetDict


class Xglue_utils(object):
    def __init__(self, subdataset):
        super().__init__()
        self.guids = None
        self.label2id = None
        self.id2label = None
        self.metric_name = []
        self.label_names = None
        self.subdataset = subdataset
        self.langs = _LANGUAGES[self.subdataset]
        raw_dataset = load_dataset('xglue.py', name=subdataset, cache_dir='./data/xglue')
        valid_datasets = []
        for lang in self.langs:
            vaild = raw_dataset['validation.{}'.format(lang)]
            valid_datasets.append(vaild)
        combined_dataset = concatenate_datasets(valid_datasets)

        if subdataset == 'mlqa':
            def add_id(example, idx):
                example["id"] = idx
                return example

            raw_dataset['train'] = raw_dataset['train'].map(add_id, with_indices=True)
            combined_dataset = combined_dataset.map(add_id, with_indices=True)

        self.raw_dataset = DatasetDict({'train': raw_dataset['train'], 'validation': combined_dataset})
        self.task_type = Task.TokenClassification if self.subdataset in ['ner', 'pos'] \
            else Task.QuestionAnswering if self.subdataset in ['mlqa'] \
            else Task.CausalLM if self.subdataset in ['qg', 'ntg'] else Task.SequenceClassification
        self.num_labels = 0

    def get_tokenizer_dataset(self, tokenizer, max_length: int):
        if self.subdataset == 'ner':
            tokenized_datasets = self.get_ner_tokenizer_dataset(tokenizer=tokenizer, max_length=max_length)
        elif self.subdataset == 'pos':
            tokenized_datasets = self.get_pos_tokenizer_dataset(tokenizer=tokenizer, max_length=max_length)
        elif self.subdataset == 'mlqa':
            tokenized_datasets = self.get_mlqa_tokenizer_dataset(tokenizer=tokenizer, max_length=max_length)
        elif self.subdataset == 'nc':
            tokenized_datasets = self.get_nc_tokenizer_dataset(tokenizer=tokenizer, max_length=max_length)
        elif self.subdataset == 'xnli':
            tokenized_datasets = self.get_xnli_tokenizer_dataset(tokenizer=tokenizer, max_length=max_length)
        elif self.subdataset == 'paws-x':
            tokenized_datasets = self.get_paws_x_tokenizer_dataset(tokenizer=tokenizer, max_length=max_length)
        elif self.subdataset == 'qadsm':
            tokenized_datasets = self.get_qadsm_tokenizer_dataset(tokenizer=tokenizer, max_length=max_length)
        elif self.subdataset == 'wpr':
            tokenized_datasets = self.get_wpr_tokenizer_dataset(tokenizer=tokenizer, max_length=max_length)
        elif self.subdataset == 'qam':
            tokenized_datasets = self.get_qam_tokenizer_dataset(tokenizer=tokenizer, max_length=max_length)
        elif self.subdataset == 'qg':
            tokenized_datasets = self.get_qg_tokenizer_dataset(tokenizer=tokenizer, max_length=max_length)
        elif self.subdataset == 'ntg':
            tokenized_datasets = self.get_ntg_tokenizer_dataset(tokenizer=tokenizer, max_length=max_length)
        else:
            exit('No subdataset {} erro get_tokenizer_dataset'.format(self.subdataset))
        return tokenized_datasets

    def get_ner_tokenizer_dataset(self, tokenizer, max_length: int):
        ner_feature = self.raw_dataset["train"].features["ner"]

        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples["words"], truncation=True, is_split_into_words=True, padding="max_length",
                max_length=max_length
            )
            all_labels = examples["ner"]
            new_labels = []
            for i, labels in enumerate(all_labels):
                word_ids = tokenized_inputs.word_ids(i)
                new_labels.append(align_labels_with_tokens(labels, word_ids))

            tokenized_inputs["labels"] = new_labels
            return tokenized_inputs

        def align_labels_with_tokens(labels, word_ids):
            new_labels = []
            current_word = None
            for word_id in word_ids:
                if word_id != current_word:
                    # Start of a new word!
                    current_word = word_id
                    label = -100 if word_id is None else labels[word_id]
                    new_labels.append(label)
                elif word_id is None:
                    # Special token
                    new_labels.append(-100)
                else:
                    # Same word as previous token
                    label = labels[word_id]
                    # If the label is B-XXX we change it to I-XXX
                    if label % 2 == 1:
                        label += 1
                    new_labels.append(label)

            return new_labels

        tokenized_datasets_t = self.raw_dataset["train"].map(tokenize_and_align_labels, batched=True,
                                                             remove_columns=self.raw_dataset["train"].column_names)
        tokenized_datasets_v = self.raw_dataset["validation"].map(tokenize_and_align_labels, batched=True,
                                                                  remove_columns=self.raw_dataset[
                                                                      "validation"].column_names)
        tokenized_datasets = DatasetDict({"train": tokenized_datasets_t, "validation": tokenized_datasets_v})
        self.label_names = ner_feature.feature.names
        self.id2label = {i: label for i, label in enumerate(self.label_names)}
        self.label2id = {v: k for k, v in self.id2label.items()}
        return tokenized_datasets

    def get_pos_tokenizer_dataset(self, tokenizer, max_length: int):
        ner_feature = self.raw_dataset["train"].features["pos"]

        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples["words"], truncation=True, is_split_into_words=True, padding="max_length",
                max_length=max_length
            )
            all_labels = examples["pos"]
            new_labels = []
            for i, labels in enumerate(all_labels):
                word_ids = tokenized_inputs.word_ids(i)
                new_labels.append(align_labels_with_tokens(labels, word_ids))

            tokenized_inputs["labels"] = new_labels
            return tokenized_inputs

        def align_labels_with_tokens(labels, word_ids):
            new_labels = []
            current_word = None
            for word_id in word_ids:
                if word_id != current_word:
                    # Start of a new word!
                    current_word = word_id
                    label = -100 if word_id is None else labels[word_id]
                    new_labels.append(label)
                elif word_id is None:
                    # Special token
                    new_labels.append(-100)
                else:
                    # Same word as previous token
                    label = labels[word_id]
                    # If the label is B-XXX we change it to I-XXX
                    if label % 2 == 1:
                        label += 1
                    new_labels.append(label)

            return new_labels

        tokenized_datasets_t = self.raw_dataset["train"].map(tokenize_and_align_labels, batched=True,
                                                             remove_columns=self.raw_dataset["train"].column_names)
        tokenized_datasets_v = self.raw_dataset["validation"].map(tokenize_and_align_labels, batched=True,
                                                                  remove_columns=self.raw_dataset[
                                                                      "validation"].column_names)
        tokenized_datasets = DatasetDict({"train": tokenized_datasets_t, "validation": tokenized_datasets_v})
        self.label_names = ner_feature.feature.names
        self.id2label = {i: label for i, label in enumerate(self.label_names)}
        self.label2id = {v: k for k, v in self.id2label.items()}
        return tokenized_datasets

    def get_mlqa_tokenizer_dataset(self, tokenizer, max_length: int):
        def preprocess_training_examples(examples):
            questions = [q.strip() for q in examples["question"]]
            inputs = tokenizer(
                questions,
                examples["context"],
                max_length=max_length,
                truncation="only_second",
                stride=64,
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
                while sequence_ids[idx] == 1:
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

        train_dataset = self.raw_dataset["train"].map(
            preprocess_training_examples,
            batched=True,
            remove_columns=self.raw_dataset["train"].column_names,
        )

        def preprocess_validation_examples(examples):
            questions = [q.strip() for q in examples["question"]]
            inputs = tokenizer(
                questions,
                examples["context"],
                max_length=max_length,
                truncation="only_second",
                stride=1,
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

        validation_dataset = self.raw_dataset["validation"].map(
            preprocess_validation_examples,
            batched=True,
            remove_columns=self.raw_dataset["validation"].column_names,
        )
        tokenized_datasets = DatasetDict({"train": train_dataset, "validation": validation_dataset})
        return tokenized_datasets

    def get_nc_tokenizer_dataset(self, tokenizer, max_length: int):
        def preprocess_data(examples):
            return tokenizer(examples["news_title"], examples["news_body"], padding="max_length", truncation=True,
                             max_length=max_length, return_tensors="pt")

        self.raw_dataset['train'] = self.raw_dataset['train'].rename_column("news_category", "label")
        self.raw_dataset['validation'] = self.raw_dataset['validation'].rename_column("news_category", "label")
        tokenized_datasets = self.raw_dataset.map(preprocess_data, batched=True)
        return tokenized_datasets

    def get_xnli_tokenizer_dataset(self, tokenizer, max_length: int):
        def preprocess_data(examples):
            return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True,
                             max_length=max_length, return_tensors="pt")

        tokenized_datasets = self.raw_dataset.map(preprocess_data, batched=True)
        return tokenized_datasets

    def get_paws_x_tokenizer_dataset(self, tokenizer, max_length: int):
        def preprocess_data(examples):
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True,
                             max_length=max_length, return_tensors="pt")

        tokenized_datasets = self.raw_dataset.map(preprocess_data, batched=True)
        return tokenized_datasets

    def get_qadsm_tokenizer_dataset(self, tokenizer, max_length: int):
        def preprocess_data(examples):
            inputs = tokenizer(examples["query"], examples["ad_title"], examples["ad_description"],
                               padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"],
                    "labels": examples["relevance_label"]}

        tokenized_datasets = self.raw_dataset.map(preprocess_data, batched=True)
        return tokenized_datasets

    def get_wpr_tokenizer_dataset(self, tokenizer, max_length: int):
        def preprocess_data(examples):
            inputs = tokenizer(examples["query"], examples["web_page_title"], examples["web_page_snippet"],
                               padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")

            return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"],
                    "labels": examples["relavance_label"]}

        tokenized_datasets = self.raw_dataset.map(preprocess_data, batched=True)
        return tokenized_datasets

    def get_qam_tokenizer_dataset(self, tokenizer, max_length: int):
        def preprocess_data(examples):
            inputs = tokenizer(examples["question"], examples["answer"], padding="max_length", truncation=True,
                               max_length=max_length,
                               return_tensors="pt", add_special_tokens=True)
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": examples["label"]
            }

        tokenized_datasets = self.raw_dataset.map(preprocess_data, batched=True)
        return tokenized_datasets

    def get_qg_tokenizer_dataset(self, tokenizer, max_length: int):
        def preprocess_data(examples):
            return tokenizer(examples["answer_passage"], examples["question"], padding="max_length", truncation=True,
                             max_length=max_length, return_tensors="pt")

        tokenized_datasets = self.raw_dataset.map(preprocess_data, batched=True)
        return tokenized_datasets

    def get_ntg_tokenizer_dataset(self, tokenizer, max_length: int):
        def preprocess_data(element):
            outputs = tokenizer(
                element["news_body"],
                truncation=True,
                max_length=max_length,
                return_overflowing_tokens=True,
                return_length=True,
            )
            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                if length == max_length:
                    input_batch.append(input_ids)
            return {"input_ids": input_batch}

        tokenized_datasets = self.raw_dataset.map(preprocess_data, batched=True,
                                                  remove_columns=self.raw_dataset["train"].column_names)
        return tokenized_datasets

    def get_label_list(self):
        return self.label_names

    def get_label_number(self):
        if self.subdataset == 'ner':
            self.num_labels = [self.id2label, self.label2id]
        elif self.subdataset == 'pos':
            self.num_labels = [self.id2label, self.label2id]
        elif self.subdataset == 'mlqa':
            self.num_labels = [0]
        elif self.subdataset == 'nc':
            self.num_labels = [10]
        elif self.subdataset == 'xnli':
            self.num_labels = [3]
        elif self.subdataset == 'paws-x':
            self.num_labels = [2]
        elif self.subdataset == 'qadsm':
            self.num_labels = [2]
        elif self.subdataset == 'wpr':
            self.num_labels = [5]
        elif self.subdataset == 'qam':
            self.num_labels = [2]
        elif self.subdataset == 'qg':
            self.num_labels = [0]
        elif self.subdataset == 'ntg':
            self.num_labels = [0]
        else:
            exit('No subdataset {} erro in get_label_number'.format(self.subdataset))
        return self.num_labels

    def get_metric(self):
        if self.subdataset in ["ner"]:
            self.metric_name = ["f1_m"]
        elif self.subdataset in ["mlqa"]:
            self.metric_name = ["exact_match", "f1"]
        elif self.subdataset in ["nc", "paws-x", "qam", 'qadsm', 'xnli']:
            self.metric_name = ["accuracy"]
        elif self.subdataset in ['pos']:
            self.metric_name = ["precision"]
        elif self.subdataset in ['wpr']:
            self.metric_name = ["nDCG"]
        elif self.subdataset in ['qg', 'ntg']:
            self.metric_name = ["BLEU-4"]
        else:
            exit('No subdataset {} erro in get_metric'.format(self.subdataset))
        return self.metric_name

    def get_raw_dataset_validation(self):
        return self.raw_dataset['validation']

    def get_wpr_grid(self):
        guids = None
        if self.subdataset == 'wpr':
            validation = self.raw_dataset['validation']
            last_query = ""
            guids = []
            guid = 0
            for item in validation['query']:
                if not item == last_query and len(last_query) > 0:
                    guid += 1
                last_query = item
                guids.append(guid)
        self.guids = guids
        return guids
