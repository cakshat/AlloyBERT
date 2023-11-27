import torch
from tqdm import tqdm


def train_pt(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss, lr = 0.0, 0.0

    for batch in tqdm(dataloader):
        ids = batch['ids'].to(device)
        attention_mask = batch['mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        output = model(ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss

        total_loss += loss.item()
        lr += optimizer.param_groups[0]['lr']

        loss.backward()
        optimizer.step()
        scheduler.step()

    return (total_loss / len(dataloader)), (lr / len(dataloader))


def validate_pt(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(dataloader):
        ids = batch['ids'].to(device)
        attention_mask = batch['mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.inference_mode():
            output = model(ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_ft(model, dataloader, optimizer, criterion, scheduler, device):
    model.train()
    total_loss, lr = 0.0, 0.0

    for batch in tqdm(dataloader):
        ids = batch['ids'].to(device)
        attention_mask = batch['mask'].to(device)
        target = batch['target'].to(device)

        optimizer.zero_grad()

        output = model(ids, attention_mask=attention_mask).squeeze()
        loss = criterion(output, target)

        total_loss += loss.item()
        lr += optimizer.param_groups[0]['lr']

        loss.backward()
        optimizer.step()
        scheduler.step()

    return (total_loss / len(dataloader)), (lr / len(dataloader))


def validate_ft(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_error = 0.0, 0.0

    for batch in tqdm(dataloader):
        ids = batch['ids'].to(device)
        attention_mask = batch['mask'].to(device)
        target = batch['target'].to(device)

        with torch.inference_mode():
            output = model(ids, attention_mask=attention_mask).squeeze()
            loss = criterion(output, target)

        total_loss += loss.item()
        total_error += torch.mean(torch.abs(target - output))

    return (total_loss / len(dataloader)), (total_error / len(dataloader))


def load_pretrained(model, pretrained_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path)['model_state_dict']

    matched_keys = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model.load_state_dict(matched_keys, strict=False)
    print('Pretrained Checkpoint Loaded')
