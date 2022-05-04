# %%
import numpy as np
import pandas as pd

import json
from pathlib import Path


# %%
# 환경 설정. [경로 생성, 데이터 로드]
data_root = '/opt/ml/input/data'
save_root = 'dataset/dkt'

train_path = save_root / Path('train')
test_path = save_root / Path('test')

train_path.mkdir(exist_ok=True, parents=True)
test_path.mkdir(exist_ok=True, parents=True)

train_data = pd.read_csv(data_root + '/train_data.csv')
test_data  = pd.read_csv(data_root + '/test_data.csv')

data = pd.concat([train_data, test_data])

# %%
# 데이터 전처리. [중복 데이터 삭제, test.csv에서 -1 삭제, train 데이터에서 test 데이터 삭제]
data.drop_duplicates(subset = ["userID", "assessmentItemID"],
                     keep = "last", inplace = True)

data_submission = data.copy()
target = test_data[test_data.answerCode == -1]
target.answerCode = 0.5
data  = data[data.answerCode>=0].copy()

eval_data = data.copy()
eval_data.drop_duplicates(subset = ["userID"],
                     keep = "last", inplace = True)

data.drop(index=eval_data.index, inplace=True, errors='ignore')

# %%
# 제출을 위한 데이터들
trg = test_data[test_data.answerCode == -1]['userID']
indices = data_submission['userID'].isin(trg)
submission = data_submission[indices].reset_index(drop=True)
answer_index = submission.drop_duplicates('userID', keep='last').index
answer_index.to_series().to_csv(f'{save_root}/answer_loc.txt', index=False, header=False)

#%%
# 데이터 encoding 함수.[RecBole에 자체적인 함수가 있는 것으로 파악되어 삭제 예정.]
def look_up_table(items):
    table = sorted(set(items))
    N = len(table)
    dict_ind = {v:i for i,v in enumerate(table)}
    return N, dict_ind

def stack(first_idx, table):
    return {k:v+first_idx for k, v in table.items()}

# %% encode cols
# 데이터 encoding 결과 저장.
_, userid_2_index = look_up_table(data.userID)
_, itemid_2_index = look_up_table(data.assessmentItemID)
_, testid_2_index = look_up_table(data.testId)
_, tagid_2_index = look_up_table(data.KnowledgeTag)

encode_table = {
    "userid":userid_2_index,
    "itemid":itemid_2_index,
    "testid":testid_2_index,
    "tagid":tagid_2_index,
}
with open(f'{save_root}/encode_table.json', "w") as json_file:
    json.dump(encode_table, json_file)

# %%
# train 데이터 encoding
print("Processing Start")
data['userID'] = data['userID'].map(lambda x: userid_2_index[x])
data['assessmentItemID'] = data['assessmentItemID'].map(lambda x: itemid_2_index[x])
data['testId'] = data['testId'].map(lambda x: testid_2_index[x])
data['Timestamp'] = data['Timestamp'].map(lambda x: pd.Timestamp(x).timestamp()).astype('int')
data['KnowledgeTag'] = data['KnowledgeTag'].map(lambda x: tagid_2_index[x])
print("Processing Complete")

# %%
# test 데이터 encoding
print("Processing Start")
eval_data['userID'] = eval_data['userID'].map(lambda x: userid_2_index[x])
eval_data['assessmentItemID'] = eval_data['assessmentItemID'].map(lambda x: itemid_2_index[x])
eval_data['testId'] = eval_data['testId'].map(lambda x: testid_2_index[x])
eval_data['Timestamp'] = eval_data['Timestamp'].map(lambda x: pd.Timestamp(x).timestamp()).astype('int')
eval_data['KnowledgeTag'] = eval_data['KnowledgeTag'].map(lambda x: tagid_2_index[x])
print("Processing Complete")

# %%
# submission 데이터 encoding
print("Processing Start")
submission['userID'] = submission['userID'].map(lambda x: userid_2_index[x])
submission['assessmentItemID'] = submission['assessmentItemID'].map(lambda x: itemid_2_index[x])
submission['testId'] = submission['testId'].map(lambda x: testid_2_index[x])
submission['Timestamp'] = submission['Timestamp'].map(lambda x: pd.Timestamp(x).timestamp()).astype('int')
submission['KnowledgeTag'] = submission['KnowledgeTag'].map(lambda x: tagid_2_index[x])
print("Processing Complete")

# %%
# target 데이터 encoding
print("Processing Start")
target['userID'] = target['userID'].map(lambda x: userid_2_index[x])
target['assessmentItemID'] = target['assessmentItemID'].map(lambda x: itemid_2_index[x])
target['testId'] = target['testId'].map(lambda x: testid_2_index[x])
target['Timestamp'] = target['Timestamp'].map(lambda x: pd.Timestamp(x).timestamp()).astype('int')
target['KnowledgeTag'] = target['KnowledgeTag'].map(lambda x: tagid_2_index[x])
print("Processing Complete")

# %%
# 실제 파일을 만드는 함수
def dump(dataframe:pd.DataFrame, cols:list, type_ls:list, dataset_name:str, extension:str='inter', drop_duplicates_col:str=None):
    print(f"{dataset_name}.{extension}\t Dump Start")
    df_inter = dataframe[cols]
    if drop_duplicates_col:
        df_inter.drop_duplicates(drop_duplicates_col, inplace=True)
    df_inter.columns = [f'{k}:{v}' for k, v in zip(df_inter.columns, type_ls)]
    df_inter.to_csv(f'{save_root}/{dataset_name}/{dataset_name}.{extension}', index=False, sep ='\t')
    print(f"{dataset_name}.{extension}\t Dump Complete")

# %%
# train datset 생성
dump(
    data, 
    cols=['userID', 'assessmentItemID', 'answerCode', 'Timestamp'], 
    type_ls=['token', 'token', 'float', 'float'], 
    dataset_name='train', extension='inter', 
    drop_duplicates_col=None
    )

dump(
    data, 
    cols=['userID'], 
    type_ls=['token'], 
    dataset_name='train', extension='user', 
    drop_duplicates_col='userID'
    )

dump(
    data, 
    cols=['assessmentItemID', 'testId', 'KnowledgeTag'], 
    type_ls=['token', 'token', 'token'], 
    dataset_name='train', extension='item', 
    drop_duplicates_col='assessmentItemID'
    )


# %%
# test datset 생성
dump(
    eval_data, 
    cols=['userID', 'assessmentItemID', 'answerCode', 'Timestamp'], 
    type_ls=['token', 'token', 'float', 'float'], 
    dataset_name='test', extension='inter', 
    drop_duplicates_col=None
    )

dump(
    eval_data, 
    cols=['userID'], 
    type_ls=['token'], 
    dataset_name='test', extension='user', 
    drop_duplicates_col='userID'
    )

dump(
    eval_data, 
    cols=['assessmentItemID', 'testId', 'KnowledgeTag'], 
    type_ls=['token', 'token', 'token'], 
    dataset_name='test', extension='item', 
    drop_duplicates_col='assessmentItemID'
    )

# %%
# submission datset 생성
dump(
    submission, 
    cols=['userID', 'assessmentItemID', 'answerCode', 'Timestamp'], 
    type_ls=['token', 'token', 'float', 'float'], 
    dataset_name='submission', extension='inter', 
    drop_duplicates_col=None
    )

dump(
    submission, 
    cols=['userID'], 
    type_ls=['token'], 
    dataset_name='submission', extension='user', 
    drop_duplicates_col='userID'
    )

dump(
    submission, 
    cols=['assessmentItemID', 'testId', 'KnowledgeTag'], 
    type_ls=['token', 'token', 'token'], 
    dataset_name='submission', extension='item', 
    drop_duplicates_col='assessmentItemID'
    )
# %%
# target datset 생성
dump(
    target, 
    cols=['userID', 'assessmentItemID', 'answerCode', 'Timestamp'], 
    type_ls=['token', 'token', 'float', 'float'], 
    dataset_name='target', extension='inter', 
    drop_duplicates_col=None
    )

dump(
    target, 
    cols=['userID'], 
    type_ls=['token'], 
    dataset_name='target', extension='user', 
    drop_duplicates_col='userID'
    )

dump(
    target, 
    cols=['assessmentItemID', 'testId', 'KnowledgeTag'], 
    type_ls=['token', 'token', 'token'], 
    dataset_name='target', extension='item', 
    drop_duplicates_col='assessmentItemID'
    )