import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model_2 import Net  # Updated to import from model_2

exer_n = 3162
knowledge_n = 102
student_n = 1709

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
epoch_n = 200

def train():
    data_loader = TrainDataLoader()
    net = Net(student_n, exer_n, knowledge_n)  # No change needed in constructor
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    print('training model...')
    affect_loss_function = nn.MSELoss()
    loss_function = nn.NLLLoss()
    for epoch in range(epoch_n):
        net.train()
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels, affects = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels, affects = [
                x.to(device) for x in [input_stu_ids, input_exer_ids, input_knowledge_embs, labels, affects]]
            optimizer.zero_grad()
            outputs = net(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output_1, affect_p = outputs
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)
            loss = loss_function(torch.log(output), labels) + affect_loss_function(affect_p, affects)
            loss.backward()
            optimizer.step()
            net.apply_clipper()
            running_loss += loss.item()
            if batch_count % 200 == 199:
                running_loss = 0.0

        rmse, auc = validate(net, epoch)
        save_snapshot(net, f'model_2/model_epoch_{epoch + 1}')

def validate(model, epoch):
    data_loader = ValTestDataLoader('validation')
    net = Net(student_n, exer_n, knowledge_n, num_heads=6)  # Assuming additional configuration for heads
    print('validating model...')
    data_loader.reset()
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    batch_count = 0  # Initialize batch_count here
    pred_all, label_all = [], []
    
    while not data_loader.is_end():
        batch_count += 1
        inputs = data_loader.next_batch()
        inputs = [x.to(device) for x in inputs[:-2]] + [x.to(device) for x in inputs[-2:]]
        output, _ = net(*inputs[:3])
        output = output.view(-1)
        labels = inputs[3]
        
        for i, label in enumerate(labels):
            correct_count += (label == (output[i] > 0.5).long()).item()
        exer_count += len(labels)
        pred_all.extend(output.to('cpu').tolist())
        label_all.extend(labels.to('cpu').tolist())

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    accuracy = correct_count / exer_count
    rmse = np.sqrt(np.mean((label_all - pred_all)**2))
    auc = roc_auc_score(label_all, pred_all)
    
    print(f'epoch= {epoch}, accuracy= {accuracy:.6f}, rmse= {rmse:.6f}, auc= {auc:.6f}')
    
    # Write validation results to file
    with open('result/model_val.txt', 'a', encoding='utf8') as f:
        f.write(f'epoch= {epoch}, accuracy= {accuracy:.6f}, rmse= {rmse:.6f}, auc= {auc:.6f}\n')

    # Ensure the function returns the calculated rmse and auc
    return rmse, auc

def save_snapshot(model, filename):
    torch.save(model.state_dict(), filename)

if __name__ == '__main__':
    if len(sys.argv) != 1:
        if (len(sys.argv) != 3) or ((sys.argv[1] != 'cpu') and ('cuda:' not in sys.argv[1])) or (not sys.argv[2].isdigit()):
            print('Usage: python train.py {device} {epoch}\nExample: python train.py cuda:0 70')
            exit(1)
        else:
            device = torch.device(sys.argv[1])
            epoch_n = int(sys.argv[2])
    train()
