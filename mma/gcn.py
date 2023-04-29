
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_regression.Alchemy_dataset import TencentAlchemyDataset
from torch_geometric.nn import GCNConv, Set2Set
from torch_geometric.data import DataLoader

import pandas as pd
#from validate import get_results, get_valid_results, get_valid_targets
import sys
# import wandb
from tqdm import tqdm

epoch = 20
# wandb.init(
#       # Set the project where this run will be logged
#       project="alchemy", 
#       # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
#       name=f'gcn_sep_{epoch}', 
#       # Track hyperparameters and run metadata
#       config={
#       "architecture": "GCN",
#       "epochs": epoch,
#       })

train_dataset = TencentAlchemyDataset(root='./data-bin', mode='dev').shuffle()
valid_dataset = TencentAlchemyDataset(root='./data-bin', mode='valid')

print(valid_dataset.data, valid_dataset.slices)


valid_loader = DataLoader(valid_dataset, batch_size=64)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


    
valid_targets = get_valid_targets()

exit()
class GCN(torch.nn.Module):
    def __init__(self,
                 node_input_dim=15,
                 output_dim=12,
                 node_hidden_dim=64,
                 num_step_prop=6,
                 num_step_set2set=6):
        super(GCN, self).__init__()
        self.num_step_prop = num_step_prop
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.conv = GCNConv(node_hidden_dim, node_hidden_dim, cached=False)        
        self.set2set = Set2Set(node_hidden_dim, processing_steps=num_step_set2set)
        self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        self.lin2 = nn.Linear(node_hidden_dim, output_dim)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        
        for i in range(self.num_step_prop):
            out = F.relu(self.conv(out, data.edge_index))
   
        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
                      
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(node_input_dim=train_dataset.num_features).to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch, valid_loader, valid_targets):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        y_model = model(data)
        loss = F.mse_loss(y_model, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    
    valid_preds = test(valid_loader)
    val_mse, val_mae = get_valid_results(valid_preds, valid_targets)

    train_loss = loss_all / len(train_loader.dataset)

    log_dict = {"Training Loss": train_loss}

    log_dict_mse = {f'Property {i} Validation MSE': val_mse[i] for i in range(len(val_mse))}
    log_dict_mae = {f'Property {i} Validation MAE': val_mae[i] for i in range(len(val_mae))}

    log_dict.update(log_dict_mae)
    log_dict.update(log_dict_mse)


    wandb.log(log_dict)
    print(f"Epoch: {epoch}, Training Loss: {train_loss}, Validation MSE: {val_mse}, Validation MAE: {val_mae}")
        
    return train_loss


def test(loader):
    model.eval()

    targets = dict()
    for data in loader:
        data = data.to(device)
        y_pred = model(data)
        #print(f"test example {data.y=}")
        #print(f"{data.__dict__=}")
        #break
        for i in range(len(data.y)):
            targets[data.y[i].item()] = y_pred[i].tolist()
    
    return targets

# epoch = 20
print("training...")
for epoch in tqdm(range(epoch)):
    loss = train(epoch, valid_loader, valid_targets)
    #wandb.log({"Train Loss": loss})

    print('Epoch: {:03d}, Loss: {:.7f}'.format(epoch, loss))

targets = test(valid_loader)
df_targets = pd.DataFrame.from_dict(targets, orient="index", columns=['property_%d' % x for x in range(12)])
df_targets.sort_index(inplace=True)
df_targets.to_csv('targets.csv', index_label='gdb_idx')
# model_name = 'gcn_epochs{epoch}}'
# get_results(model_name=model_name)