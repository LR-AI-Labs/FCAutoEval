import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import random
import time
import os
from tqdm import tqdm
from transformers import AutoModel, BitsAndBytesConfig
import math
import numpy as np
import pandas as pd
import sys
import json
import requests
from geminiAPI import evaluate_query, evaluate_response

def read_json(path) -> list:
    datas = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            datas.append(data)
    return data
def read_excel(path):
    
    # path = 'LLaMA-Factory/QCdata/TestReport_VinMotion_Action_Phase3_28032025 1.xlsx'
    #create empty dataframe
    df = pd.DataFrame()
    num_sheets = pd.ExcelFile(path).sheet_names[1:]
    for id in num_sheets:
        sheet = pd.read_excel(path, sheet_name=id, index_col=0, header=0)
        df = pd.concat([df, sheet])

    
    df = df[['title', 'custom_nlp_sample', 'custom_nlp_expected_dialog', 'custom_nlp_expected_intent']]
    #rename these coulmns
    #'custom_nlp_sample' --> query
    #'custom_nlp_expected_dialog' --> response 
    #'custom_nlp_expected_intent' --> function
    df = df.rename(columns={
        'title': 'message',
        'custom_nlp_sample': 'query',
        'custom_nlp_expected_dialog': 'response',
        'custom_nlp_expected_intent': 'function'
    })
    #shuffle df
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def get_answer(function: str = None, query = None): #get_answer by function and query 
    return '''ở Sapa, khi dùng lẩu gà đen, bạn sẽ được phục vụ một nồi nước được chế biến thuốc bắc, bên cạnh đó là nấm, rau tươi và gà đen để nhúng lẩu, thích hợp cho những ... Jan 26, 2024 ... ... các khách sạn Sapa phổ biến được du khách ưa chuộng tại đây nhé: Sa Pa Township. Tiết kiệm 3%. Khách sạn DeLaSol Sapa. •. 9.2/10. 930.150 VND. Sapa là địa điểm du lịch hấp dẫn, được nhiều du khách ưa chuộng. Bạn sẽ ... Đào Sapa thường được tìm thấy ở các khu chợ Sapa: Chợ đêm Sapa: Trung tâm ... Feb 18, 2023 ... Măng đen đắng là một trong những loại rau rừng phổ biến vào mùa mưa của dân tộc Tày tại Sa Pa. Nguyên liệu này thường được dùng để làm nem, mầm ... May 21, 2024 ... Quả thanh mai Sapa không chỉ được ưa chuộng để ăn tươi ... Mèn mén là món đặc sản Sapa thường được tìm thấy ở các phiên chợ của người Mông. Nov 19, 2024 ... Cốn sủi là một trong những đặc sản Sapa được yêu thích nhất, đặc biệt trong những ngày se lạnh. ... tại Sapa. Đây là một món đặc sản được ... Dec 29, 2023 ... Cơm lam thường là đặc sản của những vùng núi và ở Sapa cũng vậy. ... Chẳm chéo – một gia vị đặc sản Sapa làm quà rất được ưa chuộng. 4 ... Oct 12, 2022 ... Vì vậy, chúng thường xuất hiện ở vùng biển của các nước ôn đới ... số vùng có khí hậu thích hợp như Sa Pa và Đà Lạt. Nhiều nhà hàng cá ... Nov 15, 2023 ... Thịt trâu gác bếp luôn là món ăn khoái khẩu của những người dân tộc vùng cao tại Sapa ... Được ưa chuộng không chỉ vì hương vị ngon miệng mà còn ... Sep 24, 2024 ...'''

def convert_to_fc(tool_call):
    '''
    Convert raw_string to function name and query
    '''
    prefix = '<tool_call>'
    suffix = '</tool_call>'
    if prefix in tool_call and suffix in '<tool_call>':
        tool_call = tool_call[len(prefix):-len(suffix)]
    tool_call = json.loads(tool_call)
    return tool_call["name"], tool_call["arguments"]["query"]




from prompt import system_prompt
print('your system prompt:\n', system_prompt)
from tools import tools
print('your tools:\n', tools)
# system_prompt = old_system_prompt


# path_model = '/home/nlp/LLaMA-Factory/saves/minicpmo_function_calling_2/checkpoint-4608'
if __name__ == '__main__':
    #first argument is path to the model you want to evaluate
    if len(sys.argv) > 1:
        path_model = str(sys.argv[1])
    if len(sys.argv) > 2:
        gt_path =  str(sys.argv[2])
    quant_config = None
    # if len(sys.argv) > 2 and str(sys.argv[2]).lower().strip() == 'true':
    #     quant_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=getattr(torch, "float16"),
    #     bnb_4bit_use_double_quant=False,
    # )
    model = AutoModel.from_pretrained(
        path_model,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16,
        # init_vision=True,
        init_vision=False,
        init_audio=True,
        init_tts=False,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        device_map='cuda:0',
        # custom_modules=custom_modules,
    )

    tokenizer = AutoTokenizer.from_pretrained(path_model, trust_remote_code=True)

    score = 0
    cnt_function = 0
    cnt_query = 0
    cnt_response = 0
    path_save = 'answer.jsonl'
    
    ground_truth_datas = read_excel(gt_path)

    #current tools
    
    # tools = '[{"name": "get_recipe_details", "description": "Lấy chi tiết công thức nấu ăn", "parameters": {"type": "object", "properties": {"recipe_name": {"type": "string", "description": "Tên công thức nấu ăn, ví dụ: Gà nướng mật ong, Bánh sinh nhật"}}, "required": ["recipe_name"]}}, {"name": "get_recipe_feedback", "description": "Lấy phản hồi từ người dùng khác về công thức đã gửi", "parameters": {"type": "object", "properties": {"user_id": {"type": "string", "description": "ID người dùng gửi công thức"}, "recipe_name": {"type": "string", "description": "Tên công thức cần xem phản hồi"}}, "required": ["user_id", "recipe_name"]}}, {"name": "get_ingredient_suggestions", "description": "Lấy gợi ý nguyên liệu thay thế", "parameters": {"type": "object", "properties": {"ingredient": {"type": "string", "description": "Nguyên liệu cần thay thế, ví dụ: gà, bột mì"}}, "required": ["ingredient"]}}, {"name": "search_recipes_by_dietary_restrictions", "description": "Tìm kiếm công thức phù hợp với chế độ ăn", "parameters": {"type": "object", "properties": {"dietary_type": {"type": "string", "description": "Loại chế độ ăn, ví dụ: chay, không lactose, gluten-free"}}, "required": ["dietary_type"]}}, {"name": "get_saved_recipes", "description": "Lấy danh sách công thức đã lưu của người dùng", "parameters": {"type": "object", "properties": {"user_id": {"type": "string", "description": "ID người dùng"}}, "required": ["user_id"]}}, {"name": "get_recipe_suggestions", "description": "Lấy gợi ý công thức nấu ăn dựa trên loại món ăn", "parameters": {"type": "object", "properties": {"meal_type": {"type": "string", "description": "Loại món ăn, ví dụ: món chính, món tráng miệng, món khai vị"}}, "required": ["meal_type"]}}, {"name": "submit_recipe", "description": "Nhận và lưu trữ công thức mới từ người dùng", "parameters": {"type": "object", "properties": {"user_id": {"type": "string", "description": "ID người dùng gửi công thức"}, "recipe_details": {"type": "object", "description": "Thông tin công thức bao gồm tên, nguyên liệu, cách làm"}}, "required": ["user_id", "recipe_details"]}}, {"name": "save_recipe_to_account", "description": "Lưu công thức vào tài khoản người dùng", "parameters": {"type": "object", "properties": {"user_id": {"type": "string", "description": "ID người dùng"}, "recipe_name": {"type": "string", "description": "Tên công thức cần lưu"}}, "required": ["user_id", "recipe_name"]}}]'
    tools = json.loads(tools)



    for i in range(len(ground_truth_datas)):
        message = ground_truth_datas.iloc[i]['message']
        query = ground_truth_datas.iloc[i]['query']
        response = ground_truth_datas.iloc[i]['response']
        function = ground_truth_datas.iloc[i]['function']
        msgs =[{
            'role': "system",
            'content': system_prompt
        },
        {
            'role': "user",
            'content': message
        }
        ]
        # token = tokenizer.apply_chat_template(msgs, tools, tokenize=False, add_generation_prompt=True, return_dict = False, return_tensors="pt")
        # print(token) #uncomment this code to test the template
        ans = model.chat(
            msgs = msgs,
            # **token,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            # use_tts_template=True,
            tools = tools, #use this argument to make model can response tool
            do_sample=True,
            temperature=0.3,
            top_p=0.95,
            top_k=50,
            num_return_sequences=1,
            # pad_token_id=tokenizer.eos_token_id
        )
        msgs.append({
            'role': 'assistant',
            'content': ans
        })
        # print(ans) #uncomment to read this answer
        # break
        if '<tool_call>' not in ans:
            continue
        pred_function, pred_query = convert_to_fc(ans)
        if pred_function == function:
            cnt_function += 1
        else:
            continue
        if evaluate_query(query, pred_query):
            cnt_query += 1
        else:
            continue
        #function to get answer
        observation = get_answer(function, query)
        msgs.append(
            {
                'role': 'tool',
                'content': observation
            }
        )
        pred_response = model.chat(
            msgs = msgs,
            # **token,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            # use_tts_template=True,
            tools = tools,
            do_sample=True,
            temperature=0.3,
            top_p=0.95,
            top_k=50,
            num_return_sequences=1,
            # pad_token_id=tokenizer.eos_token_id
        )
        if evaluate_response(response, pred_response):
            cnt_response += 1

        if i % 100 ==  0:
            print(f'Function accuracy for now: {cnt_function/(i + 1)}')
            print(f'Query accuracy for now: {cnt_query/(i + 1)}')
            print(f'Response accuracy for now: {cnt_response/(i + 1)}')
        with open(path_save, 'a', encoding='utf-8') as f:   
            f.write(json.dumps({'id': i, 'command': message, 'function': function, 'query': query, 'response': response}, ensure_ascii=False) + '\n')
    print(f'Function accuracy: {cnt_function/(i + 1)}')
    print(f'Query accuracy: {cnt_query/(i + 1)}')
    print(f'Response accuracy: {cnt_response/(i + 1)}')

