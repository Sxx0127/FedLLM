from enum import Enum


class Task(Enum):
    SequenceClassification = 1
    TokenClassification = 2
    QuestionAnswering = 3
    CausalLM = 4