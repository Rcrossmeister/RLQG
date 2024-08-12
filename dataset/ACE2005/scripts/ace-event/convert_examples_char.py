from os import path, makedirs
import json
import collections

input_dir = "./data/processed-data/default-settings/json"
output_dir = "./data/converted-data"

if not path.exists(output_dir):
    makedirs(output_dir)

for fold in ["train", "dev", "test"]:
    g_convert = open(path.join(output_dir, fold + "_convert.json"), "w")
    with open(path.join(input_dir, fold + ".json"), "r") as g:
        for line in g:
            line = json.loads(line)
            sentences = line["sentences"]
            sent_str = line["sent_str"]
            offset = line["offsets"]
            ners = line["ner"]
            relations = line["relations"]
            events = line["events"]
            sentence_start = line["sentence_start"]
            sentence_start_char = line["sentence_start_char"]
            doc_key = line["doc_key"]
            assert len(sentence_start) == len(ners) == len(relations) == len(events) == len(sentence_start) == len(sentence_start_char) == len(offset) == len(sent_str)

            for sentence, sent_text, char_offset, ner, relation, event, s_start, s_start_c in zip(sentences, sent_str, offset, ners, relations, events, sentence_start, sentence_start_char):
                sentence_annotated = collections.OrderedDict()
                sentence_annotated["sentence"] = sent_text.replace("\n", ' ').strip()
                sentence_annotated["tokens"] = sentence
                sentence_annotated["s_start"] = s_start_c
                sentence_annotated["ner"] = ner
                sentence_annotated["char_offset"] = [one_offset - s_start_c for one_offset in char_offset]
                cur_trigger_labe = ['O'] * len(sentence)

                assert len(char_offset) == len(sentence)
                for one_entity in sentence_annotated["ner"]:
                    ner_s = char_offset[one_entity[0] - s_start] - s_start_c
                    ner_e = ner_s + len(one_entity[3]) - 1
                    one_entity[0] = ner_s
                    one_entity[1] = ner_e
                    assert sentence_annotated["sentence"][ner_s:ner_e+1] == one_entity[3]
                sentence_annotated["relation"] = relation
                for one_relation in sentence_annotated["relation"]:
                    arg1_s = char_offset[one_relation[0] - s_start] - s_start_c
                    arg1_e = arg1_s + len(one_relation[4]) - 1
                    arg2_s = char_offset[one_relation[2] - s_start] - s_start_c
                    arg2_e = arg2_s + len(one_relation[5]) - 1
                    one_relation[0] = arg1_s
                    one_relation[1] = arg1_e
                    one_relation[2] = arg2_s
                    one_relation[3] = arg2_e
                    assert sentence_annotated["sentence"][arg1_s:arg1_e+1] == one_relation[4]
                    assert sentence_annotated["sentence"][arg2_s:arg2_e+1] == one_relation[5]
                sentence_annotated["event"] = event
                for one_event in sentence_annotated["event"]:

                    event_type = one_event[0][1]
                    print(one_event[0], len(sentence))
                    for token_index in range(one_event[0][0] - s_start, one_event[0][0] - s_start + 1):
                        assert cur_trigger_labe[token_index] == 'O'
                        if token_index == one_event[0][0] - s_start:
                            cur_trigger_labe[token_index] = 'B-%s' % event_type
                        else:
                            cur_trigger_labe[token_index] = 'I-%s' % event_type
                    t_s = char_offset[one_event[0][0] - s_start] - s_start_c
                    t_e = t_s + len(one_event[0][2]) - 1
                    one_event[0] = [t_s, t_e] + one_event[0][1:]
                    assert sentence_annotated["sentence"][t_s:t_e+1] == one_event[0][3]

                    for one_arg in one_event[1:]:
                        arg_s = char_offset[one_arg[0] - s_start] - s_start_c
                        arg_e = arg_s + len(one_arg[3]) - 1
                        one_arg[0] = arg_s
                        one_arg[1] = arg_e
                        assert sentence_annotated["sentence"][arg_s:arg_e+1] == one_arg[3]
                sentence_annotated["trigger_label"] = cur_trigger_labe
                g_convert.write(json.dumps(sentence_annotated, default=int) + "\n")