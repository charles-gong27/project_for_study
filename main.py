import torch
import torchsnooper

import numpy as np
from torchvision import transforms
import torchvision.models as  models
from torch.utils.data import DataLoader
import torch.optim as optim
from train import train_model, test
from utils import plot_history
import os
import torch.utils.model_zoo as model_zoo
import utils
from dataset import MyDataset
import numpy as np
olderr = np.seterr(all='ignore')

def main(data_dir, model_name, num, lr, epochs, batch_size=16, download_data=False, save_results=False):
    """
    Main function

    Args:
        data_dir: directory to download Pascal VOC data
        model_name: resnet18, resnet34 or resnet50
        num: model_num for file management purposes (can be any postive integer. Your results stored will have this number as suffix)
        lr: initial learning rate list [lr for resnet_backbone, lr for resnet_fc]
        epochs: number of training epochs
        batch_size: batch size. Default=16
        download_data: Boolean. If true will download the entire 2012 pascal VOC data as tar to the specified data_dir.
        Set this to True only the first time you run it, and then set to False. Default False
        save_results: Store results (boolean). Default False

    Returns:
        test-time loss and average precision

    Example way of running this function:
        if __name__ == '__main__':
            main('../data/', "resnet34", num=1, lr = [1.5e-4, 5e-2], epochs = 15, batch_size=16, download_data=False, save_results=True)
    """

    #torch.cuda.set_device(0)

    model_dir = os.path.join("./models", model_name)
    #model_dir='./models/'+model_name
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)


    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
        'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
        'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',



    }

    model_collections_dict = {
        "resnet18": models.resnet18(),
        "resnet34": models.resnet34(),
        "resnet50": models.resnet50(),
        "densenet121": models.densenet121(),
        "densenet161": models.densenet161(),
        "densenet169": models.densenet169(),
        "densenet201": models.densenet201(),

    }

    # Initialize cuda parameters
    use_cuda = torch.cuda.is_available()
    np.random.seed(2020)
    torch.manual_seed(2020)
    device = torch.device("cuda" if use_cuda else "cpu")

    print("Available device = ", device)
    model = model_collections_dict[model_name]
    model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    '''                                                                    densenet需要加
    state_dict = model_zoo.load_url(model_urls[model_name])
    from collections import OrderedDict
    # 初始化一个空 dict
    new_state_dict = OrderedDict()
    # 修改 key，没有module字段则需要不上，如果有，则需要修改为 module.features
    for k, v in state_dict.items():
        if 'norm.1' in k:
            k = k.replace('norm.1','norm1')
        if'norm.2'in k:
            k = k.replace('norm.2', 'norm2')
        if 'conv.1' in k:
            k = k.replace('conv.1', 'conv1')
        if 'conv.2' in k:
            k = k.replace('conv.2', 'conv2')
        new_state_dict[k] = v
    # 加载修改后的新参数dict文件
    model.load_state_dict(new_state_dict)
    '''

    model.load_state_dict(model_zoo.load_url(model_urls[model_name]))

    '''                                                                     squeezenet改输出特征数3
    model.classifier[1] = torch.nn.Conv2d(512, 11, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = 11
    '''
    '''                                                                     densenet改输出特征数
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_ftrs, 40)
    '''
    #'''                                                                     resnet改输出特征数
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 40)                                                                                        #数据集时需改动
    #'''
    '''
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,40)
    '''
    model.to(device)

    optimizer = optim.SGD([
        {'params': list(model.parameters())[:-1], 'lr': lr[0], 'momentum': 0.9},
        {'params': list(model.parameters())[-1], 'lr': lr[1], 'momentum': 0.9}
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 12, eta_min=0, last_epoch=-1)

    # Imagnet values
    mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

    #    mean=[0.485, 0.456, 0.406]
    #    std=[0.229, 0.224, 0.225]

    transformations = transforms.Compose([transforms.Resize((300, 300)),

                                          transforms.RandomChoice([
                                              transforms.ColorJitter(brightness=(0.80, 1.20)),
                                              transforms.RandomGrayscale(p=0.25)
                                          ]),
                                          transforms.RandomHorizontalFlip(p=0.25),
                                          transforms.RandomRotation(25),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=std),
                                          ])

    transformations_valid = transforms.Compose([transforms.Resize(330),
                                                transforms.CenterCrop(300),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=mean, std=std),
                                                ])

    root = os.getcwd() + '/data/stanford40/'  # 调用图像
    # Create train dataloader
    dataset_train= MyDataset(txt=root + 'trainwordst.txt', transform=transformations)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=2, shuffle=True)
    # Create validation dataloader
    dataset_valid = MyDataset(txt=root + 'testwordst.txt', transform=transformations)

    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, num_workers=2)
    # Load the best weights before testing
    weights_file_path = os.path.join(model_dir, "model-{}.pth".format(num))
    if os.path.isfile(weights_file_path):
        print("Loading best weights")
        model.load_state_dict(torch.load(weights_file_path))

    log_file = open(os.path.join(model_dir, "log-{}.txt".format(num)), "w+")
    log_file.write("----------Experiment {} - {}-----------\n".format(num, model_name))
    log_file.write("transformations == {}\n".format(transformations.__str__()))
    trn_hist, val_hist= train_model(model, device, optimizer, scheduler, train_loader,valid_loader, model_dir, num,
                                     epochs, log_file)
    torch.cuda.empty_cache()

    plot_history(trn_hist[0], val_hist[0], "Loss", os.path.join(model_dir, "loss-{}".format(num)))
    plot_history(trn_hist[1], val_hist[1], "Accuracy", os.path.join(model_dir, "accuracy-{}".format(num)))

    log_file.close()

    # ---------------Test your model here---------------------------------------
    # Load the best weights before testing
    print("Evaluating model on test set")
    print("Loading best weights")

    model.load_state_dict(torch.load(weights_file_path))
    transformations_test = transforms.Compose([transforms.Resize(330),
                                               transforms.FiveCrop(300),
                                               transforms.Lambda(lambda crops: torch.stack(
                                                   [transforms.ToTensor()(crop) for crop in crops])),
                                               transforms.Lambda(lambda crops: torch.stack(
                                                   [transforms.Normalize(mean=mean, std=std)(crop) for crop in crops])),
                                               ])

    dataset_test = MyDataset(txt=root + 'testwordst.txt', transform=transformations_test)

    test_loader = DataLoader(dataset_test, batch_size=3, num_workers=0, shuffle=False)                                                          #int(batch_size / 5)

    if save_results:
        loss, ap, scores, gt = test(model, device, test_loader, returnAllScores=True)

        gt_path, scores_path, scores_with_gt_path = os.path.join(model_dir, "gt-{}.csv".format(num)), os.path.join(
            model_dir, "scores-{}.csv".format(num)), os.path.join(model_dir, "scores_wth_gt-{}.csv".format(num))

        utils.save_results(test_loader.dataset.imgs, gt, utils.classes, gt_path)
        utils.save_results(test_loader.dataset.imgs, scores, utils.classes, scores_path)
        utils.append_gt(gt_path, scores_path, scores_with_gt_path)

        utils.get_classification_accuracy(gt_path, scores_path,
                                          os.path.join(model_dir, "clf_vs_threshold-{}.png".format(num)))

        return loss, ap

    else:
        loss, ap = test(model, device, test_loader, returnAllScores=False)

        return loss, ap


# Execute main function here
if __name__ == '__main__':
    main('../data/', "resnet34", num=1, lr=[1.5e-4, 5e-2], epochs=10, batch_size=16, download_data=True,
         save_results=True)

#benchset resnet定为16 3，densenet定为2 1， wideresnet2 1，两个轻量级16 3