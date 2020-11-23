
import sys
import torch
import torch.nn.functional as F


from utils.progbar import Progbar
import os
DEBUG_MODE = getattr(sys, 'gettrace', None)() is not None

def stat_cuda(msg):
    print('--', msg)
    if torch.cuda.is_available():
        print('allocated: %dM, max allocated: %dM, cached: %dM, max cached: %dM' % (
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_cached() / 1024 / 1024,
        torch.cuda.max_memory_cached() / 1024 / 1024
        ))
    else:
        print('Torch cuda is not available')



def train_epoch(model, train_loader, train_writer, optimizer, log_interval, epoch, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    progbar = Progbar(len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
      with torch.autograd.detect_anomaly():
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.)


        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output_dict = model(data)
        output, memory_loss, total_memory = output_dict['out'], output_dict['memory_loss'], output_dict['total_memory']
        perform_loss, output_sample = criterion(output,target)
        #memory_loss = memory_loss / 50000
        loss = perform_loss+memory_loss
        loss.backward()
        optimizer.step()


        iter = (epoch - 1) * len(train_loader) + batch_idx
        pred = output_sample.argmax(dim=1, keepdim=True)
        acc = pred.eq(target.view_as(pred)).float().mean().item()
        progbar_txt = [("Loss",loss.item()),("SampleAcc", acc), ("ML",memory_loss.item()), ("PL",perform_loss.item()), ("TM", total_memory.item()) ]


        if batch_idx % log_interval == 0:
            train_writer.add_scalar("Training/Loss", loss.item(), iter)
            train_writer.add_scalar("Training/MemoryLoss",memory_loss.item(), iter)
            train_writer.add_scalar("Training/PerformLoss", perform_loss.item(), iter)
            train_writer.add_scalar("Training/SampleAccuracy", acc, iter)
            train_writer.add_scalar("Training/TotalMemory", total_memory.item(), iter)


        progbar.add(1, progbar_txt)



    av_acc =  progbar._values["SampleAcc"][0]/progbar._values["SampleAcc"][1]
    av_loss =  progbar._values["Loss"][0]/progbar._values["Loss"][1]
    return av_loss,av_acc


def test(model, test_loader, tb_writer=None):
    model.eval()
    test_loss = 0
    correct = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    aggregate_tb_summaries = {}

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc =  correct / len(test_loader.dataset)
    return test_loss, test_acc

def plot_weights(model, epoch = 0):

    folder_to_save = 'weights_hists'
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)

    model.module.plot_weights(name_to_save = folder_to_save+'/EP'+str(epoch))



if __name__ == '__main__':

    # profiling
    pass