import torch
import wandb
from datetime import datetime
import yaml
import os
import shutil
from data.dataloader import load_data
from model.network import create_model, cri_opt_sch
from model.utils import train_pt, validate_pt, train_ft, validate_ft

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')


def train_model_pt():
    print(f'{"="*30}{"TRAINING":^20}{"="*30}')

    best_loss = torch.inf
    for epoch in range(config['epochs']):
        train_loss, lr = train_pt(model, train_data_loader, optimizer, scheduler, device)
        print(f'Epoch {epoch+1}/{config["epochs"]} - Train Loss: {train_loss}\tLR: {lr}')
        val_loss = validate_pt(model, val_data_loader, device)
        print(f'Epoch {epoch+1}/{config["epochs"]} - Validation Loss: {val_loss}\n')
        if not config['debug']:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': lr
            })

        if val_loss <= best_loss and not config['debug']:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': lr
            }, f'{save_dir}/model.pt')
            print('Model Saved\n')
    wandb.finish()


def train_model_ft():
    print(f'{"="*30}{"TRAINING":^20}{"="*30}')

    best_loss = torch.inf
    for epoch in range(config['epochs']):
        train_loss, lr = train_ft(model, train_data_loader, optimizer, criterion, scheduler, device)
        print(f'Epoch {epoch+1}/{config["epochs"]} - Train Loss: {train_loss}\tLR: {lr}')
        val_loss, val_mae = validate_ft(model, val_data_loader, criterion, device)
        print(f'Epoch {epoch+1}/{config["epochs"]} - Validation Loss: {val_loss}\tValidation MAE: {val_mae}\n')
        if not config['debug']:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'lr': lr
            })

        if val_loss <= best_loss and not config['debug']:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'mae': val_mae,
                'lr': lr
            }, f'{save_dir}/model.pt')
            print('Model Saved\n')
    wandb.finish()


config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
config['device'] = device

model = create_model(config)
train_data_loader, val_data_loader = load_data(config)
criterion, optimizer, scheduler = cri_opt_sch(config, model)

if not config['debug']:
    run_name = f'{config["stage"]}-{datetime.now().strftime("%m%d_%H%M")}'
    wandb.init(project='AlloyBERT', name=run_name)

    save_dir = f'./checkpoints/{config["stage"]}/{run_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.copy('./config.yaml', f'{save_dir}/config.yaml')
    shutil.copy('./model/network.py', f'{save_dir}/network.py')

if config["stage"] == 'pretrain':
    train_model_pt()
if config["stage"] == 'finetune':
    train_model_ft()
