from tqdm import tqdm
import torch
import gc
import os
from utils import get_ap_score
import numpy as np
from utils import dealit

def train_model(model, device, optimizer, scheduler, train_loader, valid_loader, save_dir, model_num, epochs, log_file):
    """
    Train a deep neural network model

    Args:
        model: pytorch model object
        device: cuda or cpu
        optimizer: pytorch optimizer object
        scheduler: learning rate scheduler object that wraps the optimizer
        train_dataloader: training  image dataloader
        valid_dataloader: validation image dataloader
        save_dir: Location to save model weights, plots and log_file
        epochs: number of training epochs
        log_file: text file instance to record training and validation history

    Returns:
        Training history and Validation history (loss and average precision)
    """

    import cbowkai
    from cbowkai import modelll, make_context_vector, idx_to_word, word_to_idx, nomprebab



    tr_loss, tr_map = [], []
    val_loss, val_map = [], []
    best_val_map = 0.0

    # Each epoch has a training and validation phase
    for epoch in range(epochs):
        print("-------Epoch {}----------".format(epoch + 1))
        log_file.write("Epoch {} >>".format(epoch + 1))


        for phase in ['train', 'valid']:
            running_loss = 0.0
            running_ap = 0.0

            criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
            m = torch.nn.Sigmoid()

            if phase == 'train':
                model.train(True)  # Set model to training mode

                for data, target, senten in tqdm(train_loader):
                    # print(data)
                    target = dealit(target)
                    data = data.to(device)

                    sentenlist=[]
                    for tpp in senten:
                        tpi=tpp.split()
                        context=[tpi[0],tpi[1]]
                        context_vector = make_context_vector(context, word_to_idx)
                        nll_prob = modelll(context_vector)
                        changeprob = nomprebab(nll_prob)
                        sentenlist.append(changeprob)
                    outputtt=torch.Tensor.cuda(torch.Tensor(sentenlist))

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    output = model(data)

                    output=output*0.9+outputtt*0.1

                    loss = criterion(output, target)
                    # Get metrics here
                    running_loss += loss
                    running_ap += get_ap_score(torch.Tensor.cpu(target).detach().numpy(),
                                               torch.Tensor.cpu(m(output)).detach().numpy())

                    # Backpropagate the system the determine the gradients
                    loss.backward()

                    # Update the paramteres of the model
                    optimizer.step()

                    # clear variables
                    del data, target, output
                    gc.collect()
                    torch.cuda.empty_cache()

                    # print("loss = ", running_loss)

                num_samples = float(len(train_loader.dataset))
                tr_loss_ = running_loss / num_samples
                tr_map_ = running_ap / num_samples
                print(running_loss,running_ap,num_samples)
                print('train_loss: {:.4f}, train_avg_precision:{:.3f}'.format(
                    tr_loss_, tr_map_))

                log_file.write('train_loss: {:.4f}, train_avg_precision:{:.3f}, '.format(
                    tr_loss_, tr_map_))

                # Append the values to global arrays
                tr_loss.append(tr_loss_), tr_map.append(tr_map_)


            else:

                model.train(False)  # Set model to evaluate mode
                # torch.no_grad is for memory savings
                with torch.no_grad():
                    for data, target, senten in tqdm(valid_loader):
                        target = dealit(target)
                        data = data.to(device)

                        sentenlist = []
                        for tpp in senten:
                            tpi = tpp.split()
                            context = [tpi[0], tpi[1]]
                            context_vector = make_context_vector(context, word_to_idx)
                            nll_prob = modelll(context_vector)
                            changeprob = nomprebab(nll_prob)
                            sentenlist.append(changeprob)
                        outputtt = torch.Tensor.cuda(torch.Tensor(sentenlist))

                        # zero the parameter gradients
                        output = model(data)

                        output = output * 0.9 + outputtt * 0.1

                        loss = criterion(output, target)

                        running_loss += loss  # sum up batch loss
                        running_ap += get_ap_score(torch.Tensor.cpu(target).detach().numpy(),
                                                   torch.Tensor.cpu(m(output)).detach().numpy())

                        del data, target, output
                        gc.collect()
                        torch.cuda.empty_cache()

                    num_samples = float(len(valid_loader.dataset))
                    val_loss_ = running_loss.item() / num_samples
                    val_map_ = running_ap / num_samples

                    # Append the values to global arrays
                    val_loss.append(val_loss_), val_map.append(val_map_)

                    print('val_loss: {:.4f}, val_avg_precision:{:.3f}'.format(
                        val_loss_, val_map_))

                    log_file.write('val_loss: {:.4f}, val_avg_precision:{:.3f}\n'.format(
                        val_loss_, val_map_))

                    # Save model using val_acc
                    if val_map_ >= best_val_map:
                        best_val_map = val_map_
                        log_file.write("saving best weights...\n")
                        torch.save(model.state_dict(), os.path.join(save_dir, "model-{}.pth".format(model_num)))

        scheduler.step()

    return ([tr_loss, tr_map],[val_loss, val_map])


def test(model, device, test_loader, returnAllScores=False):
    """
    Evaluate a deep neural network model

    Args:
        model: pytorch model object
        device: cuda or cpu
        test_dataloader: test image dataloader
        returnAllScores: If true addtionally return all confidence scores and ground truth

    Returns:
        test loss and average precision. If returnAllScores = True, check Args
    """
    import cbowkai
    from cbowkai import modelll, make_context_vector, idx_to_word, word_to_idx, nomprebab

    model.train(False)

    running_loss = 0
    running_ap = 0

    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    m = torch.nn.Sigmoid()

    if returnAllScores == True:
        all_scores = np.empty((0, 40), float)
        ground_scores = np.empty((0, 40), float)

    with torch.no_grad():
        for data, target, senten in tqdm(test_loader):
            # print(data.size(), target.size())
            target = dealit(target)
            data = data.to(device)
            bs, ncrops, c, h, w = data.size()


            sentenlist = []
            for tpp in senten:
                tpi = tpp.split()
                context = [tpi[0], tpi[1]]
                context_vector = make_context_vector(context, word_to_idx)
                nll_prob = modelll(context_vector)
                changeprob = nomprebab(nll_prob)
                sentenlist.append(changeprob)
            outputtt = torch.Tensor.cuda(torch.Tensor(sentenlist))


            output = model(data.view(-1, c, h, w))

            output = output.view(bs, ncrops, -1).mean(1)

            output = output * 0.9 + outputtt * 0.1

            loss = criterion(output, target)

            running_loss += loss  # sum up batch loss
            running_ap += get_ap_score(torch.Tensor.cpu(target).detach().numpy(),
                                       torch.Tensor.cpu(m(output)).detach().numpy())

            #print(m(output),target)

            if returnAllScores == True:
                all_scores = np.append(all_scores, torch.Tensor.cpu(m(output)).detach().numpy(), axis=0)
                ground_scores = np.append(ground_scores, torch.Tensor.cpu(target).detach().numpy(), axis=0)

            del data, target, output
            gc.collect()
            torch.cuda.empty_cache()

    num_samples = float(len(test_loader.dataset))
    avg_test_loss = running_loss.item() / num_samples
    test_map = running_ap / num_samples

    print('test_loss: {:.4f}, test_avg_precision:{:.3f}'.format(
        avg_test_loss, test_map))

    f = open('test_result.txt', 'w')
    f.write('test loss:' + str(avg_test_loss) + '          ' + 'test_ap:' + str(test_map))
    f.close()

    if returnAllScores == False:
        return avg_test_loss, running_ap

    return avg_test_loss, running_ap, all_scores, ground_scores