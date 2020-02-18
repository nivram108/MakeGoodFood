import os
import pandas as pd
df = pd.read_csv('train_info.csv',header=None,names =["filename","label"])
for i,data in df.iterrows():
    dirname1 = "data7/%d/"%data['label']
    if not os.path.exists(dirname1):
        os.mkdir(dirname1)
    dirname = "data7/%d/%d/"%(data['label'],data['label'])
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    os.system("cp data2/train_set/%s %s"%(data['filename'],dirname))