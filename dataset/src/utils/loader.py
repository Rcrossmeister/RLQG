from __future__ import absolute_import, division, print_function

import json
import csv
import random
import torch
import numpy as np
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S %Z",
)

logger = logging.getLogger(__name__)


class AceExample(object):
    """
    A single training/test example for the ace dataset.
    """

    def __init__(self, sentence, events, s_start, tokens, trigger_label,
                 char_offset):
        self.sentence = sentence
        self.events = events
        self.s_start = s_start
        self.tokens = tokens
        self.trigger_label = trigger_label
        self.char_offset = char_offset

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        sentence = "event sentence: " + self.sentence
        s += sentence
        event_triggers = []
        for event in self.events:
            if event:
                event_triggers.append(
                    self.sentence[event[0][0]: event[0][1] + 1])
                event_triggers.append(str(event[0][0]))
                event_triggers.append(str(event[0][1]))
                event_triggers.append("|")
        s += " ||| event triggers: %s" % (" ".join(event_triggers))
        return s


class InputFeatures_trigger(object):
    """A single set of features of data."""

    def __init__(
            self,
            example_id,
            tokens,
            input_ids,
            segment_ids,
            mask_ids,
            labels,
            trigger_masks,
            sub_type,
    ):
        self.example_id = example_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.mask_ids = mask_ids
        self.segment_ids = segment_ids
        self.trigger_masks = trigger_masks
        self.sub_type = sub_type
        self.labels = labels


class InputFeatures_arg_t5(object):
    """A single set of features of data."""

    def __init__(
            self,
            example_id,
            input_ids,
            input_mask,
            #
            event_type,
            argument_type,
            fea_trigger_offset,
            #
            target_ids,
            target_attention_mask,
    ):
        self.example_id = example_id
        self.input_ids = input_ids
        self.input_mask = input_mask

        self.event_type = event_type
        self.argument_type = argument_type
        self.fea_trigger_offset = fea_trigger_offset

        self.target_ids = target_ids
        self.target_attention_mask = target_attention_mask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_template(input_doc):
    data_dic = {}
    main_type = None
    sub_type = None

    with open(input_doc) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        rd_index = 0
        for row in rd:
            if rd_index == 0:
                rd_index += 1
                continue
            if row[0].strip() != "":
                main_type = row[0].strip()
                data_dic[main_type] = {}
            if row[1].strip() != "":
                sub_type = row[1].strip()
                data_dic[main_type][sub_type] = {
                    "role": {},
                    "question": {},
                    "role_re": {},
                }

            arg_role = row[2]
            data_dic[main_type][sub_type]["role"][arg_role] = (
                    len(data_dic[main_type][sub_type]["role"]) + 1
            )
            data_dic[main_type][sub_type]["role_re"][
                data_dic[main_type][sub_type]["role"][arg_role]
            ] = arg_role

    with open(input_doc) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        rd_index = 0
        for row in rd:
            if rd_index == 0:
                rd_index += 1
                continue
            if row[0].strip() != "":
                main_type = row[0].strip()
            if row[1].strip() != "":
                sub_type = row[1].strip()
            question_list = row[4:]

            arg_role = row[2]
            data_dic[main_type][sub_type]["question"][arg_role] = {}
            one_question = row[3].strip()

            data_dic[main_type][sub_type]["question"][arg_role]["0"] = [
                one_question]
            data_dic[main_type][sub_type]["question"][arg_role]["0"].append(
                one_question[:-1] + " in [trigger]?"
            )

            for question in question_list:
                if question.strip() == "":
                    continue

                question = question.strip()
                found_role_index = []

                for one_role in data_dic[main_type][sub_type]["role"]:
                    role_str = "[" + one_role + "]"

                    if role_str.lower() in question.lower():
                        found_role_index.append(
                            data_dic[main_type][sub_type]["role"][one_role]
                        )

                found_role_index.sort(key=lambda x: x)
                found_role_index = [str(x) for x in found_role_index]
                string_found_role = "_".join(found_role_index)

                data_dic[main_type][sub_type]["question"][arg_role][
                    string_found_role
                ] = [question]
                data_dic[main_type][sub_type]["question"][arg_role][
                    string_found_role
                ].append(question[:-1] + " in [trigger]?")

    return data_dic


def load_sentences(path):
    examples = []
    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        line = json.loads(line)
        token_list = line["sentence"]
        s_start = int(line["s_start"])
        events = line["event"]
        if not events:
            continue

        if "tokens" in line:
            tokens = line["tokens"]
            trigger_label = line["trigger_label"]
            char_offset = line["char_offset"]

        else:
            tokens = []
            trigger_label = []
            char_offset = []

        example = AceExample(
            sentence=token_list,
            events=events,
            s_start=s_start,
            tokens=tokens,
            trigger_label=trigger_label,
            char_offset=char_offset,
        )

        examples.append(example)

    return examples


def convert_examples_to_features_sequence(
        examples, tokenizer, category_to_index, lower_case=False
):
    features = []
    all_input_ids = []
    all_segment_ids = []
    all_mask_ids = []
    all_labels = []
    all_trigger_masks = []
    feature_index = 0
    all_feature_index = []
    for example_id, example in enumerate(examples):
        raw_tokens = example.tokens
        trigger_label = example.trigger_label

        offset_category = dict()
        for t_idx, one_label in enumerate(trigger_label):
            if one_label != "O":
                offset_category[t_idx] = one_label

        tokens = []
        segment_ids = []
        mask_ids = []
        labels = []
        trigger_masks = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        mask_ids.append(1)
        labels.append(-100)
        trigger_masks.append(0)

        for i, token in enumerate(raw_tokens):
            if lower_case:
                token = token.lower()
            sub_tokens = tokenizer.tokenize(token)

            for sidx, sub_token in enumerate(sub_tokens):
                tokens.append(sub_token)
                segment_ids.append(0)
                mask_ids.append(1)

                if sidx == 0:
                    trigger_masks.append(1)
                    if i in offset_category:
                        labels.append(category_to_index[offset_category[i]])
                    else:
                        labels.append(0)
                else:
                    trigger_masks.append(0)
                    labels.append(-100)

        tokens.append("[SEP]")
        segment_ids.append(0)
        mask_ids.append(1)
        labels.append(-100)
        trigger_masks.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        all_input_ids.append(input_ids)
        all_segment_ids.append(segment_ids)
        all_mask_ids.append(mask_ids)
        all_labels.append(labels)
        all_feature_index.append(feature_index)
        all_trigger_masks.append(trigger_masks)

        one_feature = InputFeatures_trigger(
            example_id,
            tokens,
            input_ids,
            segment_ids,
            mask_ids,
            labels,
            trigger_masks,
            "sub_type",
        )
        features.append(one_feature)
        feature_index += 1

    bucket_dataset = Event_trigger_dataset(
        all_input_ids,
        all_segment_ids,
        all_mask_ids,
        all_labels,
        all_feature_index,
        all_trigger_masks,
    )
    return bucket_dataset, features


def normalize_ace_arg(arg_role, event_type):
    """
    fix inconsistent ACE annotations for argument role
    :param arg_role:
    :param event_type:
    :return:
    """
    if event_type == "Personnel.Elect" and arg_role == "Entity":
        arg_role = "Agent"
    if event_type == "Justice.Appeal" and arg_role == "Plaintiff":
        arg_role = "Defendant"
    if event_type == "Life.Die" and arg_role == "Person":
        arg_role = "Victim"
    if event_type == "Conflict.Attack" and arg_role == "Victim":
        arg_role = "Target"
    if event_type == "Conflict.Attack" and arg_role == "Agent":
        arg_role = "Attacker"
    if event_type == "Movement.Transport" and arg_role == "Victim":
        arg_role = "Artifact"
    if event_type == "Movement.Transport" and arg_role == "Place":
        arg_role = "Destination"
    return arg_role


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def my_collate_trigger_train(batch):
    list_input_ids = []
    list_token_type_ids = []
    list_attention_masks = []
    list_trigger_labels = []
    list_sentence_id = []
    list_trigger_masks = []

    max_len = 0
    for idx, sample in enumerate(batch):
        input_ids = sample[0]
        max_len = max(max_len, len(input_ids))

    for idx, sample in enumerate(batch):
        cur_len = len(sample[0])
        input_ids = sample[0] + [0] * (max_len - cur_len)
        list_input_ids.append(input_ids)

        token_type_ids = sample[1] + [0] * (max_len - cur_len)
        list_token_type_ids.append(token_type_ids)

        attention_masks = sample[2] + [0] * (max_len - cur_len)
        list_attention_masks.append(attention_masks)

        trigger_labels = sample[3] + [-100] * (max_len - cur_len)
        list_trigger_labels.append(trigger_labels)

        sentence_id = sample[4]
        list_sentence_id.append(sentence_id)

        trigger_masks = sample[5] + [0] * (max_len - cur_len)
        list_trigger_masks.append(trigger_masks)

    input_ids_tensor = torch.LongTensor(list_input_ids)
    token_type_ids_tensor = torch.LongTensor(list_token_type_ids)
    attention_masks_tensor = torch.LongTensor(list_attention_masks)
    trigger_labels_tensor = torch.LongTensor(list_trigger_labels)
    trigger_masks_tensor = torch.LongTensor(list_trigger_masks)
    sentence_id_tensor = torch.LongTensor(list_sentence_id)

    return (
        input_ids_tensor,
        trigger_labels_tensor,
        attention_masks_tensor,
        token_type_ids_tensor,
        sentence_id_tensor,
        trigger_masks_tensor,
    )


def my_collate_qg_train_t5(batch):
    pad_id = 0
    list_input_ids = []
    list_attention_masks = []
    list_output_ids = []
    list_feature_idx = []

    max_len_input = 0
    max_len_output = 0
    for idx, sample in enumerate(batch):
        input_ids = sample[0]
        output_ids = sample[2]
        max_len_input = max(max_len_input, len(input_ids))
        max_len_output = max(max_len_output, len(output_ids))

    for idx, sample in enumerate(batch):
        cur_len_input = len(sample[0])
        input_ids = sample[0] + [pad_id] * (max_len_input - cur_len_input)
        list_input_ids.append(input_ids)

        attention_mask = sample[1] + [0] * (max_len_input - cur_len_input)
        list_attention_masks.append(attention_mask)

        cur_len_output = len(sample[2])
        output_ids = sample[2] + [pad_id] * (max_len_output - cur_len_output)
        list_output_ids.append(output_ids)

        list_feature_idx.append(sample[3])

    input_ids_tensor = torch.LongTensor(list_input_ids)
    attention_masks_tensor = torch.LongTensor(list_attention_masks)
    output_ids_tensor = torch.LongTensor(list_output_ids)
    output_ids_tensor[output_ids_tensor == 0] = -100
    feature_idx_tensor = torch.LongTensor(list_feature_idx)

    return (
        input_ids_tensor,
        attention_masks_tensor,
        output_ids_tensor,
        feature_idx_tensor,
    )


def my_collate_qg_train_bart(batch):
    pad_id = 1
    list_input_ids = []
    list_attention_masks = []
    list_output_ids = []
    list_feature_idx = []
    max_len_input = 0
    max_len_output = 0
    for idx, sample in enumerate(batch):
        input_ids = sample[0]
        output_ids = sample[2]
        max_len_input = max(max_len_input, len(input_ids))
        max_len_output = max(max_len_output, len(output_ids))

    for idx, sample in enumerate(batch):
        cur_len_input = len(sample[0])
        input_ids = sample[0] + [pad_id] * (max_len_input - cur_len_input)
        list_input_ids.append(input_ids)

        attention_mask = sample[1] + [0] * (max_len_input - cur_len_input)
        list_attention_masks.append(attention_mask)

        cur_len_output = len(sample[2])
        output_ids = sample[2] + [pad_id] * (max_len_output - cur_len_output)
        list_output_ids.append(output_ids)

        list_feature_idx.append(sample[3])

    input_ids_tensor = torch.LongTensor(list_input_ids)
    attention_masks_tensor = torch.LongTensor(list_attention_masks)
    output_ids_tensor = torch.LongTensor(list_output_ids)
    output_ids_tensor[output_ids_tensor == 0] = -100
    feature_idx_tensor = torch.LongTensor(list_feature_idx)

    return (
        input_ids_tensor,
        attention_masks_tensor,
        output_ids_tensor,
        feature_idx_tensor,
    )

def retrieve_answer(all_roles, arguments):
    answers = {role: [] for role in all_roles}
    for argument in arguments:
        role = argument[2]
        answers.setdefault(role, []).append(argument[3])
    return answers

def convert_examples_to_features_qga(
        examples, query_templates, template_type, lower_case=False
):
    input_texts = []
    target_texts = []
    answer_texts = []
    feature_idx = 0
    for example_id, example in enumerate(examples):
        for event in example.events:
            event_type = event[0][2]
            main_event_type = event_type.split(".")[0]
            sub_event_type = event_type.split(".")[1]
            trigger_token = event[0][3]
            arguments = event[1:]
            all_roles = \
                list(query_templates[main_event_type][sub_event_type]["role"].keys())
            answers = retrieve_answer(all_roles, arguments)


            for target_argument_type in query_templates[main_event_type][
                sub_event_type]["question"]:
                query_dic = {}
                base_query = query_templates[main_event_type][sub_event_type][
                    "question"
                ][target_argument_type]["0"][0]
                role_mapping = \
                    query_templates[main_event_type][sub_event_type]["role"]

                query_dic["0"] = [base_query]

                # retrieve the argument roles that
                # co-occure with the target argument
                other_role_set = set()
                for arg_idx, argument in enumerate(arguments):
                    answer = argument[3]
                    raw_argument_type = argument[2]
                    normalized_argument_type = normalize_ace_arg(
                        raw_argument_type, event_type
                    )

                    # in ACE 2005 event guideline, Contact.Phone-Write
                    # should not have Place argument
                    if (
                            event_type == "Contact.Phone-Write"
                            and normalized_argument_type == "Place"
                    ):
                        continue

                    # skip the arguments with the same argument role
                    # as the target argument
                    if normalized_argument_type == target_argument_type:
                        continue
                    other_role_set.add(normalized_argument_type)
                num_args = len(other_role_set)

                # check if the target argument role mentioned
                # in the event mention example
                if_arg_mentioned = False
                for arg_idx, argument in enumerate(arguments):
                    raw_argument_type = argument[2]
                    normalized_argument_type = normalize_ace_arg(
                        raw_argument_type, event_type
                    )
                    if (
                            event_type == "Contact.Phone-Write"
                            and normalized_argument_type == "Place"
                    ):
                        continue

                    if normalized_argument_type == target_argument_type:
                        if_arg_mentioned = True

                if template_type == "dynamic":
                    query_dic = generate_questions_with_contextual_args(
                        query_dic,
                        num_args,
                        role_mapping,
                        target_argument_type,
                        arguments,
                        event_type,
                        other_role_set,
                        query_templates,
                    )

                    # pick up the template with the most contextual arguments.
                    # if the target argument role is no mentioned in the sample,
                    # pick up the basic template.
                    if not if_arg_mentioned:
                        training_query_list = query_dic["0"]
                    elif "4" in query_dic:
                        training_query_list = query_dic["4"]
                    elif "3" in query_dic:
                        training_query_list = query_dic["3"]
                    elif "2" in query_dic:
                        training_query_list = query_dic["2"]
                    elif "1" in query_dic:
                        training_query_list = query_dic["1"]
                    else:
                        training_query_list = query_dic["0"]

                    if len(training_query_list) > 1:
                        query_list = [random.choice(training_query_list)]
                    else:
                        query_list = training_query_list

                elif template_type == "annotation":
                    training_query_list = query_dic["0"]

                    query_list = training_query_list
                elif template_type == "standard":
                    training_query_list = query_dic["0"]
                    if "Where" in training_query_list[0]:
                        training_query_list[0] = training_query_list[0].replace("event", trigger_token)
                    else:
                        training_query_list[0] = training_query_list[0][:-1] + " in " + trigger_token + "?"

                    query_list = training_query_list
                elif template_type == "raw":
                    training_query_list = query_dic["0"]

                    query_list = training_query_list

                for query in query_list:
                    target_text = query + " </s>"
                    if lower_case:
                        target_text = target_text.lower()

                    assert (
                            example.sentence[
                            event[0][0]: event[0][1] + 1] == trigger_token
                    )
                    question_context = (
                            example.sentence[: event[0][0]]
                            + "* "
                            + trigger_token
                            + " *"
                            + example.sentence[event[0][1] + 1:]
                    )
                    input_text = "role: %s trigger: %s context: %s </s>" % (
                        target_argument_type.lower(),
                        trigger_token,
                        question_context,
                    )

                    answer_text = answers[target_argument_type]

                    if lower_case:
                        input_text = input_text.lower()
                        answer_text = [answer.lower() for answer in answer_text]

                    input_texts.append(input_text)
                    target_texts.append(target_text)
                    answer_texts.append(answer_text)
                    feature_idx += 1

    dataset = zip(input_texts, target_texts, answer_texts)
    return dataset


def template_fill_in(
        arguments,
        start_idx,
        num_role,
        event_type,
        other_role_set,
        found_other_arg_list,
        found_other_role_list,
        role_mapping,
        query_templates,
        target_argument_type,
        query_dic,
        threshold_arg,
):
    for sub_idx in range(start_idx + 1, len(arguments)):
        cur_argument = arguments[sub_idx]
        argument_role = cur_argument[2]
        normalized_argument_role = normalize_ace_arg(argument_role, event_type)
        if event_type == "Contact.Phone-Write" and normalized_argument_role == "Place":
            continue
        if normalized_argument_role not in other_role_set:
            continue
        cur_argument[2] = normalized_argument_role

        if cur_argument in found_other_arg_list:
            continue
        if role_mapping[normalized_argument_role] in found_other_role_list:
            continue
        else:
            found_other_arg_list.append(cur_argument)
            found_other_role_list.append(
                role_mapping[normalized_argument_role])

        if num_role - 1 > 0:
            found_other_arg_list, found_other_role_list, query_dic = template_fill_in(
                arguments,
                sub_idx,
                num_role - 1,
                event_type,
                other_role_set,
                found_other_arg_list,
                found_other_role_list,
                role_mapping,
                query_templates,
                target_argument_type,
                query_dic,
                threshold_arg,
            )
            found_other_arg_list.pop()
            found_other_role_list.pop()
        else:
            assert len(found_other_arg_list) == threshold_arg
            found_other_arg_index_list = [
                role_mapping[x[2]] for x in found_other_arg_list
            ]
            found_other_arg_index_list.sort(key=lambda x: x)
            found_other_arg_index_list = [str(x) for x in
                                          found_other_arg_index_list]
            query = query_templates[event_type.split(".")[0]][
                event_type.split(".")[1]][
                "question"
            ][target_argument_type]["_".join(found_other_arg_index_list)][0]

            for one_arg_name in found_other_arg_list:
                arg_str = one_arg_name[3]
                query = query.replace("[" + one_arg_name[2] + "]", arg_str)
            query_dic[str(threshold_arg)].append(query)
            found_other_arg_list.pop()
            found_other_role_list.pop()
    return found_other_arg_list, found_other_role_list, query_dic


def convert_examples_to_features_qa(
        examples,
        tokenizer,
        query_templates,
        nth_query,
        is_training=True,
        multi_arg=False,
        qg_model=None,
        na_format="empty",
        qg_tokenizer=None,
        transformer_name="t5-large",
        qg_length_penalty=1.0,
        lower_case=False,
        qa_inf_batch_size=16,
        qg_num_beams=4,
):
    all_input_ids = []
    all_target_ids = []

    all_mask_ids = []
    all_target_mask_ids = []
    all_features = []

    all_feature_idex = []
    feature_idx = 0
    input_text_dic = {}

    all_query_input_ids = []
    all_query_mask_ids = []
    all_query_feature_idex = []

    for example_id, example in enumerate(examples):
        for event in example.events:
            trigger_offset = event[0][0]
            event_type = event[0][2]
            main_event_type = event_type.split(".")[0]
            sub_event_type = event_type.split(".")[1]
            trigger_token = event[0][3]
            arguments = event[1:]

            for target_argument_type in query_templates[main_event_type][
                sub_event_type
            ]["question"]:
                query_dic = {}
                base_query = query_templates[main_event_type][sub_event_type][
                    "question"
                ][target_argument_type]["0"][nth_query]

                role_mapping = \
                    query_templates[main_event_type][sub_event_type]["role"]

                query_dic["0"] = [base_query]

                # retrieve the argument roles that
                # co-occure with the target argument
                other_role_set = set()
                for arg_idx, argument in enumerate(arguments):
                    raw_argument_type = argument[2]
                    normalized_argument_type = normalize_ace_arg(
                        raw_argument_type, event_type
                    )

                    # in ACE 2005 event guideline, Contact.Phone-Write
                    # should not have Place argument
                    if (
                            event_type == "Contact.Phone-Write"
                            and normalized_argument_type == "Place"
                    ):
                        continue

                    # skip the arguments with the same argument role
                    # as the target argument
                    if normalized_argument_type == target_argument_type:
                        continue
                    other_role_set.add(normalized_argument_type)
                num_args = len(other_role_set)

                no_answer = True
                answer_list = []
                for arg_idx, argument in enumerate(arguments):
                    raw_argument_type = argument[2]
                    normalized_argument_type = normalize_ace_arg(
                        raw_argument_type, event_type
                    )

                    if (
                            event_type == "Contact.Phone-Write"
                            and normalized_argument_type == "Place"
                    ):
                        continue

                    argument[2] = normalized_argument_type
                    if normalized_argument_type == target_argument_type:
                        no_answer = False

                        answer_start, answer_end = argument[0], argument[1]
                        answer_str = argument[3]
                        answer_list.append(
                            [answer_start, answer_end, answer_str])

                query_dic = generate_questions_with_contextual_args(
                    query_dic,
                    num_args,
                    role_mapping,
                    target_argument_type,
                    arguments,
                    event_type,
                    other_role_set,
                    query_templates,
                )

                # generate embeddings for output
                answer_list.sort(key=lambda x: x[0])
                if not no_answer:
                    targ_str_list = [x[2] + " " for x in answer_list]
                    target_text = "; ".join(targ_str_list)
                    target_text += "</s>"
                else:
                    if na_format == "empty":
                        target_text = "</s>"
                    else:
                        target_text = "No Answer </s>"
                if lower_case:
                    target_text = target_text.lower()
                target_encodings = tokenizer.encode_plus(target_text)
                target_ids = target_encodings["input_ids"]
                target_attention_mask = target_encodings["attention_mask"]

                if is_training:
                    for query_index in query_dic:
                        if no_answer and query_index != "0":
                            continue
                        for one_query in query_dic[query_index]:
                            # append the query with trigger information
                            input_text = (
                                    "question: %s in * %s * event? context: %s </s>"
                                    % (one_query[:-1], trigger_token,
                                       example.sentence)
                            )
                            if lower_case:
                                input_text = input_text.lower()
                            input_encodings = tokenizer.encode_plus(input_text)
                            input_ids = input_encodings["input_ids"]
                            input_mask = input_encodings["attention_mask"]

                            all_input_ids.append(input_ids)
                            all_mask_ids.append(input_mask)
                            all_target_ids.append(target_ids)
                            all_target_mask_ids.append(target_attention_mask)
                            all_feature_idex.append(feature_idx)
                            feature_idx += 1

                            one_feature = InputFeatures_arg_t5(
                                example_id,
                                input_ids,
                                input_mask,
                                event_type,
                                target_argument_type,
                                trigger_offset,
                                target_ids,
                                target_attention_mask,
                            )
                            all_features.append(one_feature)

                else:
                    assert (
                            example.sentence[
                            event[0][0]: event[0][1] + 1] == trigger_token
                    )
                    question_context = (
                            example.sentence[: event[0][0]]
                            + "* "
                            + trigger_token
                            + " *"
                            + example.sentence[event[0][1] + 1:]
                    )
                    query_input_text = "role: %s context: %s </s>" % (
                        target_argument_type.lower(),
                        question_context,
                    )
                    if lower_case:
                        query_input_text = query_input_text.lower()
                    query_input_encodings = qg_tokenizer.encode_plus(
                        query_input_text)
                    query_input_ids = query_input_encodings["input_ids"]
                    query_input_mask = query_input_encodings["attention_mask"]

                    all_query_input_ids.append(query_input_ids)
                    all_query_mask_ids.append(query_input_mask)
                    all_query_feature_idex.append(feature_idx)

                    # for _ in range(num_question):
                    input_ids, input_mask = None, None
                    all_input_ids.append(None)
                    all_mask_ids.append(None)
                    all_target_ids.append(target_ids)
                    all_target_mask_ids.append(target_attention_mask)
                    all_feature_idex.append(feature_idx)

                    input_text_dic[feature_idx] = trigger_token
                    feature_idx += 1

                    one_feature = InputFeatures_arg_t5(
                        example_id,
                        input_ids,
                        input_mask,
                        event_type,
                        target_argument_type,
                        trigger_offset,
                        target_ids,
                        target_attention_mask,
                    )
                    all_features.append(one_feature)

    qg_model = qg_model.cuda()
    query_bucket_dataset = QG_dataset_inference(
        all_query_input_ids, all_query_mask_ids, all_query_feature_idex
    )
    if transformer_name == "t5-large":
        query_dataset_loader = torch.utils.data.DataLoader(
            query_bucket_dataset,
            qa_inf_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=my_collate_qa_inference_t5,
        )
    elif transformer_name == "bart-large":
        query_dataset_loader = torch.utils.data.DataLoader(
            query_bucket_dataset,
            qa_inf_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=my_collate_qa_inference_bart,
        )

    qg_model.eval()

    for idx, batch in enumerate(query_dataset_loader):
        if is_training:
            logger.info(f"is_training: {is_training}")
        input_ids, input_mask, example_indices = batch
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        with torch.no_grad():
            outputs = qg_model.generate(
                input_ids=input_ids,
                attention_mask=input_mask,
                num_beams=qg_num_beams,
                early_stopping=False,
                # num_return_sequences=num_question,
                length_penalty=qg_length_penalty,
            )

        for i, example_index in enumerate(example_indices):
            example_index = example_index.cpu().item()
            if example_index not in input_text_dic:
                continue
            cur_feature = all_features[example_index]

            # for index_return in range(num_question):
            pre_answer = qg_tokenizer.decode(outputs[i],
                                             skip_special_tokens=True)
            # pre_answer = qg_tokenizer.decode(
            #     outputs[i * num_question + index_return],
            #     skip_special_tokens=True)
            pre_answer = pre_answer.strip()

            input_text = "question: %s in * %s * event? context: %s </s>" % (
                pre_answer[:-1],
                input_text_dic[example_index],
                examples[cur_feature.example_id].sentence,
            )
            if lower_case:
                input_text = input_text.lower()
            input_encodings = tokenizer.encode_plus(input_text)
            input_ids = input_encodings["input_ids"]
            input_mask = input_encodings["attention_mask"]

            all_features[example_index].input_ids = input_ids
            all_features[example_index].input_mask = input_mask

            all_input_ids[example_index] = input_ids
            all_mask_ids[example_index] = input_mask
            # all_features[
            #     example_index + index_return].input_ids = input_ids
            # all_features[
            #     example_index + index_return].input_mask = input_mask

            # all_input_ids[example_index + index_return] = input_ids
            # all_mask_ids[example_index + index_return] = input_mask

    bucket_dataset = QA_dataset(
        all_input_ids, all_mask_ids, all_target_ids, all_feature_idex
    )
    return bucket_dataset, all_features


def my_collate_qa_t5(batch):
    pad_id = 0
    list_input_ids = []
    list_attention_masks = []
    list_output_ids = []
    list_feature_idx = []
    max_len = 0
    max_len_target = 0
    for idx, sample in enumerate(batch):
        input_ids = sample[0]
        output_ids = sample[2]
        max_len = max(max_len, len(input_ids))
        max_len_target = max(max_len_target, len(output_ids))

    for idx, sample in enumerate(batch):
        cur_len = len(sample[0])
        input_ids = sample[0] + [pad_id] * (max_len - cur_len)
        list_input_ids.append(input_ids)

        attention_masks = sample[1] + [0] * (max_len - cur_len)
        list_attention_masks.append(attention_masks)

        cur_len_target = len(sample[2])
        output_ids = sample[2] + [pad_id] * (max_len_target - cur_len_target)
        list_output_ids.append(output_ids)

        list_feature_idx.append(sample[3])

    input_ids_tensor = torch.LongTensor(list_input_ids)
    attention_masks_tensor = torch.LongTensor(list_attention_masks)
    output_ids_tensor = torch.LongTensor(list_output_ids)
    output_ids_tensor[output_ids_tensor == 0] = -100
    feature_idx_tensor = torch.LongTensor(list_feature_idx)

    return (
        input_ids_tensor,
        attention_masks_tensor,
        output_ids_tensor,
        feature_idx_tensor,
    )


def my_collate_qa_bart(batch):
    pad_id = 1
    list_input_ids = []
    list_attention_masks = []
    list_output_ids = []
    list_feature_idx = []
    max_len = 0
    max_len_target = 0
    for idx, sample in enumerate(batch):
        input_ids = sample[0]
        output_ids = sample[2]
        max_len = max(max_len, len(input_ids))
        max_len_target = max(max_len_target, len(output_ids))
    for idx, sample in enumerate(batch):
        cur_len = len(sample[0])
        input_ids = sample[0] + [pad_id] * (max_len - cur_len)
        list_input_ids.append(input_ids)

        attention_masks = sample[1] + [0] * (max_len - cur_len)
        list_attention_masks.append(attention_masks)

        cur_len_target = len(sample[2])
        output_ids = sample[2] + [pad_id] * (max_len_target - cur_len_target)
        list_output_ids.append(output_ids)

        list_feature_idx.append(sample[3])

    input_ids_tensor = torch.LongTensor(list_input_ids)
    attention_masks_tensor = torch.LongTensor(list_attention_masks)
    output_ids_tensor = torch.LongTensor(list_output_ids)
    output_ids_tensor[output_ids_tensor == 0] = -100
    feature_idx_tensor = torch.LongTensor(list_feature_idx)

    return (
        input_ids_tensor,
        attention_masks_tensor,
        output_ids_tensor,
        feature_idx_tensor,
    )


def my_collate_qa_inference_t5(batch):
    pad_id = 0
    list_input_ids = []
    list_attention_masks = []
    list_feature_idx = []
    max_len = 0
    for idx, sample in enumerate(batch):
        input_ids = sample[0]
        max_len = max(max_len, len(input_ids))
    for idx, sample in enumerate(batch):
        cur_len = len(sample[0])
        input_ids = sample[0] + [pad_id] * (max_len - cur_len)
        list_input_ids.append(input_ids)

        attention_masks = sample[1] + [0] * (max_len - cur_len)
        list_attention_masks.append(attention_masks)

        list_feature_idx.append(sample[2])

    input_ids_tensor = torch.LongTensor(list_input_ids)
    attention_masks_tensor = torch.LongTensor(list_attention_masks)
    feature_idx_tensor = torch.LongTensor(list_feature_idx)

    return input_ids_tensor, attention_masks_tensor, feature_idx_tensor


def my_collate_qa_inference_bart(batch):
    pad_id = 1
    list_input_ids = []
    list_attention_masks = []
    list_feature_idx = []
    max_len = 0
    for idx, sample in enumerate(batch):
        input_ids = sample[0]
        max_len = max(max_len, len(input_ids))

    for idx, sample in enumerate(batch):
        cur_len = len(sample[0])
        input_ids = sample[0] + [pad_id] * (max_len - cur_len)
        list_input_ids.append(input_ids)
        attention_masks = sample[1] + [0] * (max_len - cur_len)
        list_attention_masks.append(attention_masks)
        list_feature_idx.append(sample[2])

    input_ids_tensor = torch.LongTensor(list_input_ids)
    attention_masks_tensor = torch.LongTensor(list_attention_masks)
    feature_idx_tensor = torch.LongTensor(list_feature_idx)

    return input_ids_tensor, attention_masks_tensor, feature_idx_tensor


def generate_questions_with_contextual_args(
        query_dic,
        num_args,
        role_mapping,
        target_argument_type,
        arguments,
        event_type,
        other_role_set,
        query_templates,
):
    if num_args >= 1:
        threshold_arg = 1
        query_dic[str(threshold_arg)] = []
        found_other_arg_list = []
        found_other_role_list = [role_mapping[target_argument_type]]

        (
            found_other_arg_list,
            found_other_role_list,
            query_dic,
        ) = template_fill_in(
            arguments,
            -1,
            1,
            event_type,
            other_role_set,
            found_other_arg_list,
            found_other_role_list,
            role_mapping,
            query_templates,
            target_argument_type,
            query_dic,
            threshold_arg,
        )

    if num_args >= 2:
        threshold_arg = 2
        query_dic[str(threshold_arg)] = []
        found_other_arg_list = []
        found_other_role_list = [role_mapping[target_argument_type]]

        (
            found_other_arg_list,
            found_other_role_list,
            query_dic,
        ) = template_fill_in(
            arguments,
            -1,
            2,
            event_type,
            other_role_set,
            found_other_arg_list,
            found_other_role_list,
            role_mapping,
            query_templates,
            target_argument_type,
            query_dic,
            threshold_arg,
        )

    if num_args >= 3:
        threshold_arg = 3
        query_dic[str(threshold_arg)] = []
        found_other_arg_list = []
        found_other_role_list = [role_mapping[target_argument_type]]

        (
            found_other_arg_list,
            found_other_role_list,
            query_dic,
        ) = template_fill_in(
            arguments,
            -1,
            3,
            event_type,
            other_role_set,
            found_other_arg_list,
            found_other_role_list,
            role_mapping,
            query_templates,
            target_argument_type,
            query_dic,
            threshold_arg,
        )

    if num_args >= 4:
        threshold_arg = 4
        query_dic[str(threshold_arg)] = []
        found_other_arg_list = []
        found_other_role_list = [role_mapping[target_argument_type]]

        (
            found_other_arg_list,
            found_other_role_list,
            query_dic,
        ) = template_fill_in(
            arguments,
            -1,
            4,
            event_type,
            other_role_set,
            found_other_arg_list,
            found_other_role_list,
            role_mapping,
            query_templates,
            target_argument_type,
            query_dic,
            threshold_arg,
        )
    assert num_args < 6
    return query_dic
