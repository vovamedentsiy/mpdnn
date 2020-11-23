import torch
import torchvision
from artemis.fileman.local_dir import get_local_dir, get_local_path
from artemis.general.display import CaptureStdOut

from sacred import Experiment
from tensorboardX import SummaryWriter

from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import Subset
from torchvision import datasets, transforms
import os
import sys
import numpy as np
import torch.nn.functional as F
import gc


from experiments.train_loops import train_epoch, test, plot_weights
from models.resnet_cifar import resnet20



from utils.hooks import TBHook
from utils.logging_utils import get_experiment_dir, write_error_trace

from utils.train_utils import save_state



ex = Experiment("Resnet20_Cifar10_MPDNN_U3_LOGparam_l1e-3")

DEBUG_MODE = getattr(sys, 'gettrace', None)() is not None


@ex.capture
def get_data(train_batch_size, test_batch_size):
    kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() and not DEBUG_MODE else {} # tried 3 workers, check memory
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # from yerlan idelbayev code , this seems to be for imagenet dataset!!!!!!! WTF !!!!!!????? # okay, pretrained fp model was trained with these values, but quantized model will be trained with correct normalization values!!
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # from yerlan idelbayev code
    ])

    trainset = torchvision.datasets.CIFAR10(get_local_dir("data/cifar10"), train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(get_local_dir("data/cifar10"), train=True, download=False, transform=transform_test)
    testset = torchvision.datasets.CIFAR10(get_local_dir("data/cifar10"), train=False, download=False, transform=transform_test)

    num_train = len(trainset)
    indices = list(range(num_train))
    train_idx, valid_idx = indices[:45000], indices[45000:]
    val_dataset = Subset(valset, valid_idx)
    train_dataset = Subset(trainset, train_idx)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, **kwargs)
    return trainloader, valloader, testloader


@ex.capture
def get_data_train_test(train_batch_size, test_batch_size):
    kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() and not DEBUG_MODE else {} # tried 3 workers
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(get_local_dir("data/cifar10"), train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(get_local_dir("data/cifar10"), train=False, download=False, transform=transform_test)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, **kwargs)


    return trainloader, testloader





@ex.capture
def get_model(quantize_activations, quantize_weights, use_batchnorm, model, memory_weights_constraints_flag, lambda_memory_weights_loss, memory_weights_constraints):

    if model == "resnet20":
        net = resnet20(use_batchnorm=use_batchnorm,
                       quantize_weights=quantize_weights, quantize_activations=quantize_activations,
                       memory_weights_constraints_flag = memory_weights_constraints_flag,
                       lambda_memory_weights_loss = lambda_memory_weights_loss,
                       memory_weights_constraints = memory_weights_constraints)
    else:
        raise ValueError("Model {} not known".format(model))

    return net


@ex.config
def config():
    train_batch_size = 128
    test_batch_size = 128

    lr = 0.01
    epochs = 170 # 100
    log_interval = 100
    restore_path = None 
    pretrain_path = None # use this if you want to init with pretrained FP model

    use_batchnorm = True

    quantize_activations = False
    quantize_weights = True


    act_fn = "relu"
    model = "resnet20" # "resnet20"
    schedule = "resnet" # learning rate scheduler

    memory_weights_constraints_flag = True
    lambda_memory_weights_loss = 0.001
    memory_weights_constraints = 70.





@ex.capture
def get_loss_criterion():

    def loss_criterion(output, target):
        return F.cross_entropy(output, target), output

    return loss_criterion


@ex.capture
def get_lr_scheduler(lr, schedule, optimizer):
    if schedule == "plateau":
        return ReduceLROnPlateau(optimizer, 'max', factor=0.9, patience=1, verbose=True, min_lr=0.001 * lr)

    elif schedule == "resnet":
        def _schedule(epoch):
            if epoch >= 80 and epoch < 120:
                mul = 0.1
            elif epoch >= 120:
                mul = 0.01
            else:
                mul = 1.0
            return mul
        return LambdaLR(optimizer, lr_lambda=_schedule)

    elif schedule == 'no':
        def _schedule(epoch):
            mul = 1.0
            return mul

        return LambdaLR(optimizer, lr_lambda=_schedule)

    else:
        raise NotImplementedError('ERROR LR scheduler')


@ex.capture
def configure_starting_point(lr, quantize_weights, quantize_activations, restore_path, pretrain_path, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ####################################
    # pretrain path : init. a model with pretrained FP model
    # restore path : load model trained with mpdnn and load parameters of the optimizer as well

    ####################################


    if restore_path is not None:



        print("==> Resuming from Checkpoint...")

        if torch.cuda.is_available():
            checkpoint = torch.load(restore_path)
        else:
            checkpoint = torch.load(restore_path, map_location='cpu')


        _model = {k[7:] if 'module.' in k else k: v for k, v in checkpoint["model"].items() }



        #############################################

        try:
            model.load_state_dict(_model)
        except RuntimeError as e:

            model_dict = model.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in _model.items() if k in model_dict and 'scale_premultiplier' not in k}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)

        #############################################


        model = model.to(device)


        ###################################
        flag_restore = True
        if flag_restore:
            optimizer = torch.optim.SGD(model.parameters(), lr=checkpoint['optimizer']['param_groups'][0]['lr'], momentum = 0.9)
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum = 0.9)
            optimizer.load_state_dict(checkpoint['optimizer'])
        ###################################


        epoch, best_val_acc, best_val_epoch = checkpoint["epoch"], checkpoint["best_val_acc"], checkpoint["best_epoch"]
        start_epoch = epoch + 1


    else:
        initialized_activation_quantizers = False
        need_to_initialize_activations = True
        if pretrain_path is not None:
            # strict = True
            print("==> Loading Pretrained Network")

            if torch.cuda.is_available():
                state = torch.load(pretrain_path)
            else:
                state = torch.load(pretrain_path, map_location='cpu')


            # _optimizer = state["optimizer"]
            if pretrain_path.endswith(".pt"):
                _model = {k[7:] if 'module.' in k else k: v for k, v in state["model"].items()}
            elif pretrain_path.endswith(".th"):
                _model = {k[7:] if 'module.' in k else k: v for k, v in state['state_dict'].items()}

            #############################################
            try:
                model.load_state_dict(_model)
            except RuntimeError as e:

                model_dict = model.state_dict()
                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in _model.items() if
                                   k in model_dict and 'scale_premultiplier' not in k}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                model.load_state_dict(model_dict)
            #############################################
            model = model.to(device)
            print("==> Starting from pretrained Model")

            # init weight and activ quant

        model = model.to(device)



        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum = 0.9)# , weight_decay = 0.0002) # weight decay for the baseline

        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        start_epoch = 0
        best_val_acc = -np.inf
        best_val_epoch = 0


    return model, optimizer, best_val_acc, start_epoch, best_val_epoch


@ex.automain
def cifar10(epochs, log_interval, pretrain_path, restore_path, use_batchnorm, quantize_activations, quantize_weights, _seed, _run):

    print('get_local_dir', get_local_dir("data/cifar10"))

    assert (pretrain_path is None) + (restore_path is None) > 0, "Only pretrain_path or restore_path"

    exp_dir = get_experiment_dir(ex.path, _run)

    print("Starting Experiment in {}".format(exp_dir))
    with CaptureStdOut(log_file_path=os.path.join(exp_dir, "output.txt") if not False else os.path.join(exp_dir, "val_output.txt")):
        try:
            # Data
            train_loader, test_loader = get_data_train_test()


            # Model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = get_model()

            # Configure

            model, optimizer, best_val_acc, start_epoch, best_val_epoch = configure_starting_point(model=model)
            model = model.to(device)

            # Misc
            train_writer = SummaryWriter(log_dir=exp_dir)
            hooks = TBHook(model, train_writer, start_epoch * len(train_loader), torch.cuda.device_count(), log_interval)


            scheduler = get_lr_scheduler(optimizer=optimizer)




            gc.collect()
            model = torch.nn.DataParallel(model)
            gc.collect()

            best_epoch = 0
            best_val_acc = -np.inf
            criterion = get_loss_criterion()




            if torch.cuda.is_available():
                _, test_acc = test(model, test_loader)
                print('Test acc before training ', test_acc)




            ##########################################################################################
            start_epoch = 0
            val_list = []
            for epoch in range(start_epoch, start_epoch + epochs + 1):

                ##########################################################################################

                print('EPOCH: ', epoch)

                ##########################################################################################
                print('Training')
                train_loss, train_acc = train_epoch(model=model, train_loader=train_loader, optimizer=optimizer, epoch=epoch, train_writer=train_writer,
                                                    log_interval=log_interval, criterion=criterion)
                ##########################################################################################



                train_loss_eval, train_acc_eval = test(model, train_loader)
                train_writer.add_scalar("Validation/TrainLoss", train_loss_eval, epoch * len(train_loader))
                train_writer.add_scalar("Validation/TrainAccuracy", train_acc_eval, epoch * len(train_loader))
                print("Epoch {}, Training Eval Loss: {:.4f}, Training Eval Acc: {:.4f}".format(epoch, train_loss_eval, train_acc_eval))


                val_loss, val_acc = test(model, test_loader)

                ##########################################################################################


                try:
                    scheduler.step(epoch = epoch)
                except TypeError:
                    scheduler.step()

                ##########################################################################################

                train_writer.add_scalar("Validation/Loss", val_loss, epoch * len(train_loader))
                train_writer.add_scalar("Validation/Accuracy", val_acc, epoch * len(train_loader))
                train_writer.add_scalar("Others/LearningRate", optimizer.param_groups[0]["lr"], epoch * len(train_loader))
                if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_epoch = epoch
                        save_state(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, best_val_acc=best_val_acc, best_epoch=best_epoch,
                                   save_path=os.path.join(exp_dir, "best_model.pt"))
                print("Epoch {}, Validation Loss: {:.4f},\033[1m Validation Acc: {:.4f}\033[0m , Best Val Acc: {:.4f} at EP {}".format(epoch, val_loss,
                                                                                                                                           val_acc,
                                                                                                                                           best_val_acc,
                                                                                                                                           best_epoch))

                # saving the last model
                save_state(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, best_val_acc=best_val_acc, best_epoch=best_epoch,
                           save_path=os.path.join(exp_dir, "model.pt"))


                # save all models, print real bops
                folder_to_save = 'mpdnn_models'
                if not os.path.exists(folder_to_save):
                    os.makedirs(folder_to_save)


                name_to_save = 'model_'+str(epoch)+'.pt'
                print('Epoch: ', epoch)
                print('Val ACC ', val_acc)
                if epoch % 5 == 0:
                    print('Plot weights')
                    plot_weights(model, epoch)
                    print('Saving a model ')
                    save_state(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                           best_val_acc=best_val_acc, best_epoch=best_epoch,
                           save_path=folder_to_save+'/'+name_to_save)




            
            print("Early Stopping Epoch {} with Val Acc {:.4f} ".format(best_epoch, best_val_acc))

        except Exception:
            write_error_trace(exp_dir)
            raise
