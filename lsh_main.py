import numpy as np
import pandas as pd

file_path = '/data/tapas_my/tapas_nq_retriever_large/'
file_name = 'tables.tsv'
table_file_ = pd.read_csv(file_path+file_name, delimiter='\t')

data = np.load('res18_dev_cos.npy', allow_pickle=True)

query_path = '/data/tapas_my/nq_data_dir/interactions/'
file_name = ['dev.jsonl', 'test.jsonl']
str_file_dev = pd.read_json(f'{query_path}{file_name[1]}', lines=True)
table_file = pd.read_json('/data/tapas_my/nq_data_dir/tables/tables.jsonl', lines=True)

col = []
for p in data[0]:
    tmp = []
    for i in range(len(table_file['columns'][list(table_file_[table_file_[table_file_.columns[0]] == p].index)[0]])):
        tmp.append(list(table_file['columns'][list(table_file_[table_file_[table_file_.columns[0]] == p].index)[0]][i].values())[0])
    col.append(tmp)

row = []
for i in data[0]:
    tmp = []
    for j in range(len(table_file['rows'][list(table_file_[table_file_[table_file_.columns[0]] == i].index)[0]])):
        tmp2 = []
        for z in range(len(list(table_file['rows'][list(table_file_[table_file_[table_file_.columns[0]] == i].index)[0]][j].values()))):
            for t in range(len(list(table_file['rows'][list(table_file_[table_file_[table_file_.columns[0]] == i].index)[0]][j].values())[z])):
                tmp2.append(list(list(table_file['rows'][list(table_file_[table_file_[table_file_.columns[0]] == i].index)[0]][j].values())[z][t].values())[0])
        tmp.append(tmp2)
    row.append(tmp)

print(str_file_dev['questions'][0])
find = input('what do you want to find?: ')
for da, c in zip(row, col):
    table = pd.DataFrame(da, columns=c)
    for d in da:
        if find in d:
            print('Find!!')
            print(table)
            input()
