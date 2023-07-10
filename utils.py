#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import copy
import random
import string
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset, load_from_disk

from uie.extraction.constants import BaseStructureMarker


def count_dataset(dataset, tokenizer):
    len_list = []
    target_len_list = []
    for example in dataset:
        len_list.append(len(tokenizer(example["text"])["input_ids"]))
        target_len_list.append(len(tokenizer(example["record"])["input_ids"]))
    print("total examples: ", len(dataset))
    print("len_min: ", min(len_list))
    print("len_max: ", max(len_list))
    print("len_mean: ", sum(len_list) / len(len_list))
    print("target_len_min: ", min(target_len_list))
    print("target_len_max: ", max(target_len_list))
    print("target_len_mean: ", sum(target_len_list) / len(target_len_list))
    print()


def count_datasets(datasets, tokenizer):
    print("***train dataset***")
    count_dataset(datasets["train"], tokenizer)
    print("***val dataset***")
    count_dataset(datasets["validation"], tokenizer)
    print("***test dataset***")
    count_dataset(datasets["test"], tokenizer)


def merge(example, another_example, structure_marker):
    example["tokens"] = example["tokens"] + another_example["tokens"]
    example["text"] = example["text"] + " " + another_example["text"]
    example["entity"] = example["entity"] + another_example["entity"]
    example["relation"] = example["relation"] + another_example["relation"]
    example["event"] = example["event"] + another_example["event"]
    example["spot"] = example["spot"] + another_example["spot"]
    example["asoc"] = example["asoc"] + another_example["asoc"]
    example["spot_asoc"] = example["spot_asoc"] + another_example["spot_asoc"]
    example["record"] = ' '.join([
        structure_marker.sent_start,
        " ".join(example["record"].split()[1:-1]),
        " ".join(another_example["record"].split()[1:-1]),
        structure_marker.sent_end,
    ])


def build_hard(train_dataset, sent_num=2, M=1):
    # for each training example of the main task, we randomly sample M training examples to construct M hard instances
    structure_marker = BaseStructureMarker()
    hard_train_dataset = None
    for _ in range(M):
        dataset = copy.deepcopy(train_dataset)
        data = []
        for instance in dataset:
            if len(instance["record"].split()) == 2 and instance["record"].split()[1] == structure_marker.sent_end:
                continue
            else:
                data.append(instance)
        data = {k: [r.get(k) for r in data] for k in data[0]}
        new_dataset = Dataset.from_dict(data, features=dataset.features)  # convert dict to dataset
        examples = []
        for i, example in enumerate(dataset):
            for n in range(1, sent_num):
                another_example = random.choice(new_dataset)
                merge(example, another_example, structure_marker)
            examples.append(example)
        examples = {k: [r.get(k) for r in examples] for k in examples[0]}
        dataset = Dataset.from_dict(examples, features=dataset.features)  # convert dict to dataset
        hard_train_dataset = concatenate_datasets([hard_train_dataset, dataset]) if hard_train_dataset else dataset
        
    return hard_train_dataset


def convert_spot_asoc(spot_asoc_instance, structure_marker):
    spot_instance_str_rep_list = list()
    for spot in spot_asoc_instance:
        spot_str_rep = [
            spot['label'],
            structure_marker.target_span_start,
            spot['span'],
        ]
        for asoc_label, asoc_span in spot.get('asoc', list()):
            asoc_str_rep = [
                structure_marker.span_start,
                asoc_label,
                structure_marker.target_span_start,
                asoc_span,
                structure_marker.span_end,
            ]
            spot_str_rep += [' '.join(asoc_str_rep)]
        spot_instance_str_rep_list += [' '.join([
            structure_marker.record_start,
            ' '.join(spot_str_rep),
            structure_marker.record_end,
        ])]
    target_text = ' '.join([
        structure_marker.sent_start,
        ' '.join(spot_instance_str_rep_list),
        structure_marker.sent_end,
    ])
    return target_text


def convert_spot(spot_instance, structure_marker):
    spot_str_rep_list = []
    for spot in spot_instance:
        spot_str_rep_list += [
            structure_marker.span_start,
            spot,
            structure_marker.span_end,
        ]

    target_text = ' '.join([
        structure_marker.sent_start,
        ' '.join(spot_str_rep_list),
        structure_marker.sent_end,
    ])
    return target_text


def convert_asoc(asoc_instance, structure_marker):
    asoc_str_rep_list = []
    for asoc in asoc_instance:
        asoc_str_rep_list += [
            structure_marker.span_start,
            asoc,
            structure_marker.span_end,
        ]

    target_text = ' '.join([
        structure_marker.sent_start,
        ' '.join(asoc_str_rep_list),
        structure_marker.sent_end,
    ])
    return target_text


def balance_data(data, data_args):
    empty_data = [x for x in data if x["empty"]]
    non_empty_data = [x for x in data if not x["empty"]]
    # print("len(empty_data): ", len(empty_data))
    # print("len(non_empty_data): ", len(non_empty_data))
    if len(empty_data) / len(data) > data_args.empty_ratio:
        random.shuffle(empty_data)
        num = int(len(non_empty_data) * data_args.empty_ratio / (1 - data_args.empty_ratio))
        empty_data = empty_data[:num]
        # print("len(empty_data) after downsampling: ", len(empty_data))
        data = non_empty_data + empty_data
        # random.shuffle(data)
    return data


