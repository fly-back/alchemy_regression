import pandas as pd
import numpy as np
def get_results(gen ="targets.csv", root= "data-bin/raw", mode="valid", model_name='gcn'):
    
    preds_df = pd.read_csv(gen, header=0).sort_values('gdb_idx',ascending=True).reset_index()
    true_df = pd.read_csv(root+'/'+mode+'/'+mode+'_target.csv', header=0).sort_values('gdb_idx', ascending=True).reset_index()

    preds = preds_df.values[:,1:]
    true = true_df.values[:,1:]
    mse = get_mse(preds, true)
    mae = get_mae(preds, true)

    print(f"Model name: {model_name}, MSE: {mse}, MAE: {mae}")

def get_valid_results(pred_dict, valid_dict):

    combined = [(valid_dict[idx], pred_dict[idx]) for idx in valid_dict]

    true = np.array([d[0] for d in combined])
    preds = np.array([d[1] for d in combined])  

    mse = get_mse(preds, true)
    mae = get_mae(preds, true)
    return mse, mae
    


def get_mse(preds, true):
    #print(true-preds.shape)
    return np.mean(((true-preds)**2), axis=0)

def get_mae(preds, true):
    return np.mean(np.abs(preds-true), axis=0)


def get_valid_targets(root='data-bin/raw', mode='valid'):
    targets = pd.read_csv(root + '/' + mode + '/' + mode +'_target.csv', header=0)
    target_dict = {int(row['gdb_idx']):np.array(row[1:]) for i,row in targets.iterrows()}
    return target_dict



# valid_targets = get_valid_targets()
# for v in valid_targets:
#     print(f"{v=}, \n{valid_targets[v]=}, \n{type(valid_targets[v])=}")
#     break
