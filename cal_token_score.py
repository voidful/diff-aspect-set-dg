# Import necessary libraries
import json
import statistics
from collections import defaultdict
from itertools import combinations

import nlp2
from nlgeval import NLGEval

# Initialize NLGEval object without skipthoughts and glove models, and omit the METEOR metric
nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=["METEOR"])

# Create defaultdicts for storing data and output
data_dict = defaultdict(lambda: defaultdict(dict))
output_dict = defaultdict(list)

# Define a list of answer choices
ans_list = ["a", "b", "c", "d"]

# Read the 'race_test_gold.jsonl' file and store the data in data_dict
with open('./baseline/race_test_gold.jsonl', 'r', encoding='utf8') as jsonlfile:
    for jlines in jsonlfile.readlines():
        jfile = json.loads(jlines)
        for q in jfile['questions']:
            dict_id = jfile['article'].strip() + q.strip()
            dict_id = dict_id.replace(" ", "").lower()
            ans_index = ans_list.index(jfile['answers'][0])
            answer = jfile['options'][0][ans_index]
            jfile['options'][0].pop(ans_index)
            data_dict[dict_id]['human'] = {
                'article': jfile['article'],
                'questions': jfile['questions'][0],
                'answers': answer,
                'options': jfile['options'][0]
            }

# Read the 'prev_sota.jsonl' file and store the data in data_dict
with open('./baseline/prev_sota.jsonl', 'r', encoding='utf8') as jsonlfile:
    for jlines in jsonlfile.readlines():
        jfile = json.loads(jlines)
        for q in jfile['questions']:
            dict_id = jfile['article'].strip() + q.strip()
            dict_id = dict_id.replace(" ", "").lower()
            ans_index = ans_list.index(jfile['answers'][0])
            answer = jfile['options'][0][ans_index]
            jfile['options'][0].pop(ans_index)
            data_dict[dict_id]['prev-sota'] = {
                'article': jfile['article'],
                'questions': jfile['questions'][0],
                'answers': answer,
                'options': jfile['options'][0]
            }

# Read the '8.pt_dataset_textcsv_mode_greedy_filtersim_False_predicted.csv' file and store the data in data_dict
pfile = nlp2.read_csv(f"./baseline/8.pt_dataset_textcsv_mode_greedy_filtersim_False_predicted.csv")
for i in pfile[1:]:
    c, q, a = i[0].split("</s>")
    dict_id = c.strip() + q.strip()
    dict_id = dict_id.replace(" ", "").lower()
    data_dict[dict_id]["baseline"] = {
        'article': c,
        'questions': q,
        'answers': a,
        'options': i[1].split("<s>"),
    }

# Define the evaluation targets
eval_targets = ['prev-sota', 'baseline']

# Evaluate the models and calculate metrics
for eval_target in eval_targets:
    result_dict = defaultdict(list)
    for data in data_dict.values():
        merge_dict = defaultdict(dict)
        for key, value in data.items():
            merge_dict['article'] = value['article']
            merge_dict['questions'] = value['questions']
            merge_dict['options'][key] = value['options']
            merge_dict['answer'][key] = value['answers']

    # Check if the number of keys in the 'options' dictionary is equal to the number of evaluation targets plus one
    if len((merge_dict['options'].keys())) == len(eval_targets) + 1:
        model_pred = merge_dict['options'][eval_target]
        answer = merge_dict['answer'][eval_target]
        question = merge_dict['questions']
        article = merge_dict['article']
        truth = merge_dict['options']['human']

        # Compare model prediction with human prediction
        predicted = model_pred
        target_list = [truth] * len(predicted)
        eval_metric = nlgeval.compute_metrics(ref_list=list(map(list, zip(*target_list))),  # transpose
                                              hyp_list=predicted)
        for k, v in eval_metric.items():
            result_dict[k].append(v)

        # Compare model prediction with the correct answer
        predicted = model_pred
        target_list = [answer] * len(predicted)
        eval_metric = nlgeval.compute_metrics(ref_list=list(map(list, zip(*target_list))),  # transpose
                                              hyp_list=predicted)
        for k, v in eval_metric.items():
            result_dict[k + '_answer'].append(v)

        # Compare model prediction with the article sentences
        predicted = model_pred
        target_list = [article] * len(predicted)
        eval_metric = nlgeval.compute_metrics(ref_list=list(map(list, zip(*target_list))),  # transpose
                                              hyp_list=predicted)
        for k, v in eval_metric.items():
            result_dict[k + '_article'].append(v)

        # Compare model prediction with the question
        predicted = model_pred
        target_list = [question] * len(predicted)
        eval_metric = nlgeval.compute_metrics(ref_list=list(map(list, zip(*target_list))),  # transpose
                                              hyp_list=predicted)
        for k, v in eval_metric.items():
            result_dict[k + '_question'].append(v)

        # Calculate metrics between model predictions
        internal_dict = defaultdict(list)
        for combo in combinations(model_pred, 2):
            eval_metric = nlgeval.compute_metrics(ref_list=[[combo[0]]],  # transpose
                                                  hyp_list=[combo[1]])
            for k, v in eval_metric.items():
                internal_dict[k + '_internal'].append(v)
        for k, v in internal_dict.items():
            result_dict[k].append(statistics.mean(v))

# Print the evaluation results (mean and length)
for k, v in result_dict.items():
    print(eval_target, k, statistics.mean(v), len(v))
