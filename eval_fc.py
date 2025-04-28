from geminiAPI import evaluate_response
from test_fc import read_json, read_excel
import logging
import time
import tqdm
logging.basicConfig(filename=f'output/logging_general.txt', level=logging.WARNING, format='%(message)s')
if __name__ == '__main__':
    gt_data = read_excel('/home/nlp/QC/TestReport_VinRobotic_HalfDuplex_SmokeFullScope_23042025.xlsx', '21013_Google Search')
    path = '/home/nlp/fc_autoeval_pipeline/data/general_sum.jsonl'
    data = read_json(path)
    cnt_func = 0
    cnt_sum = 0
    print('len data', len(data))
    cnt = 0
    for ds in data:
        cnt += 1
        id = int(ds['question_id'])
        groundtruth = ds['groundtruth']
        prediction = ds['prediction']
        true_summary = gt_data.iloc[id]['response']
        if groundtruth == '' or not isinstance(true_summary, str):
            logging.warning('Null data in groundtruth, or not having expcected resonse by QC. Skipping....')
            cnt -= 1
            continue
        if groundtruth == prediction:
            cnt_func+=1
        else:
            logging.warning('Different function in data:' + str(id))
            continue
        
        
        pred_summary = ds['sum']
        if evaluate_response(true_summary, pred_summary):
            cnt_sum += 1
        else:
            logging.warning('Different summary in data:' + str(id))
            logging.warning('True response: \n' + true_summary)
            logging.warning('Pred response: \n' + pred_summary)
        time.sleep(4)
        if cnt % 50 == 0:
            logging.warning(f'Accuracy for sampels {cnt}:')
            logging.warning(f'For function: {cnt_func/cnt}')
            logging.warning(f'For summary: {cnt_sum/cnt}')
    logging.warning(f'Total number of validated data: {cnt}')
    logging.warning(f'Accuracy function: {cnt_func/cnt}')
    logging.warning(f'Accuracy summary: {cnt_sum/cnt}')