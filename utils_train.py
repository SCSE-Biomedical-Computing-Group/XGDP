import torch
import torch.nn as nn
import time
import nvidia_smi
from utils_data import *

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, log_interval, return_attention_weights=False):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.MSELoss()
    avg_loss = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # output, _ = model(data)
        x, x_cell_mut, edge_index, batch_drug, edge_feat = data.x, data.target, data.edge_index.long(), data.batch, data.edge_features
        output, _ = model(x, edge_index, x_cell_mut, batch_drug, edge_feat)
        
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return sum(avg_loss)/len(avg_loss)

def predicting(model, device, loader, return_attention_weights = False):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # output, _ = model(data)
            x, x_cell_mut, edge_index, batch_drug, edge_feat = data.x, data.target, data.edge_index.long(), data.batch, data.edge_features
            if return_attention_weights:
                output, _, attn_weights = model(x, edge_index, x_cell_mut, batch_drug, edge_feat, return_attention_weights)
            else: 
                output, _ = model(x, edge_index, x_cell_mut, batch_drug, edge_feat)
        
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
            # attn_weights = attn_weights.cpu().numpy()
    torch.cuda.empty_cache()  ## no grad
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

def main(modeling, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name, br_fol, result_folder, model_folder, save_name, return_attention_weights, do_save = True):

    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)

    model_st = modeling.__name__
    dataset = 'GDSC'
    train_losses = []
    val_losses = []
    val_pearsons = []
    print('\nrunning on ', model_st + '_' + dataset )

    # processed_data_file_train = 'data/processed/' + dataset + '_train_mix'+'.pt'
    # processed_data_file_val = 'data/processed/' + dataset + '_val_mix'+'.pt'
    # processed_data_file_test = 'data/processed/' + dataset + '_test_mix'+'.pt'
    processed_data_file_train = br_fol + '/processed/' + dataset + '_train_mix'+'.pt'
    processed_data_file_val = br_fol + '/processed/' + dataset + '_val_mix'+'.pt'
    processed_data_file_test = br_fol + '/processed/' + dataset + '_test_mix'+'.pt'

    # root_folder+"root_001/processed/GDSC_train_mix.pt"

    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_val)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root=br_fol, dataset=dataset+'_train_mix')
        val_data = TestbedDataset(root=br_fol, dataset=dataset+'_val_mix')
        test_data = TestbedDataset(root=br_fol, dataset=dataset+'_test_mix')


        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
        print("CPU/GPU: ", torch.cuda.is_available())

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        print(device)
        model = modeling().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_mse = 1000
        best_pearson = 1
        best_epoch = -1
        model_file_name = 'model_' + save_name + '_' + dataset +  '.model'
        # result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'
        result_file_name = 'result_' + save_name + '_' + dataset +  '.csv'
        loss_fig_name = 'model_' + save_name + '_' + dataset + '_loss'
        pearson_fig_name = 'model_' + save_name + '_' + dataset + '_pearson'
        total_time = 0
        for epoch in range(num_epoch):
            # torch.cuda.empty_cache()
            start_time = time.time()
            print(f"epoch : {epoch+1}/{num_epoch} ")



            #
            nvidia_smi.nvmlInit()

            deviceCount = nvidia_smi.nvmlDeviceGetCount()
            for i in range(deviceCount):
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))

            nvidia_smi.nvmlShutdown()
            ######################





            train_loss = train(model, device, train_loader, optimizer, epoch+1, log_interval)
            G,P = predicting(model, device, val_loader)
            ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P)]

            if return_attention_weights:
                G_test,P_test, attn_weights = predicting(model, device, test_loader, return_attention_weights)
            else:
                G_test,P_test = predicting(model, device, test_loader)
                
            ret_test = [rmse(G_test,P_test),mse(G_test,P_test),pearson(G_test,P_test),spearman(G_test,P_test)]

            train_losses.append(train_loss)
            val_losses.append(ret[1])
            val_pearsons.append(ret[2])

            if ret[1]<best_mse:
                if (do_save):
                    torch.save(model.state_dict(), model_folder + model_file_name)
                    with open(result_folder + "val_"+ result_file_name,'w') as f:
                        f.write(','.join(map(str,ret)))
                    with open(result_folder + "test_"+ result_file_name,'w') as f:
                        f.write(','.join(map(str,ret_test)))
                    if return_attention_weights:
                        np.save(br_fol + '/Saliency/AttnWeight/' + model_st + '.npy', attn_weights)
                    
                best_epoch = epoch+1
                best_mse = ret[1]
                best_pearson = ret[2]
                print(f"ret = {ret}")
                print(f"ret_test = {ret_test}")
                print(' rmse improved at epoch ', best_epoch, '; best_mse:', best_mse,model_st,dataset)
            else:
                print(f"ret = {ret}")
                print(f"ret_test = {ret_test}")
                print(' no improvement since epoch ', best_epoch, '; best_mse, best pearson:', best_mse, best_pearson, model_st, dataset)

            total_time += time.time() - start_time
            remaining_time = (num_epoch-epoch-1)*(total_time)/(epoch+1)
            print(f"End of Epoch {epoch+1}; {int(remaining_time//3600)} hours, {int((remaining_time//60)%60)} minutes, and {int(remaining_time%60)} seconds remaining")

        draw_loss(train_losses, val_losses, result_folder + loss_fig_name)
        draw_pearson(val_pearsons, result_folder + pearson_fig_name)