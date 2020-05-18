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

    model_dir = os.path.join("./models", model_name)
    #model_dir='./models/'+model_name
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    print('------start---------')
    model_collections_dict = {
        "resnet18": models.resnet18(),
        "resnet34": models.resnet34(),
        "resnet50": models.resnet50(),
        #"densenet121": models.densenet121(),
        #"densenet161": models.densenet161(),
        #"densenet169": models.densenet169(),
        #"densenet201": models.densenet201(),
        #'squeezenet1_0': models.squeezenet1_0(),
        #"squeezenet1_1": models.squeezenet1_1(),
        #"shufflenetv2_x0.5": models.shufflenet_v2_x0_5(),
        #"vgg16": models.vgg16(),
        #"vgg19": models.vgg19(),
        #"inception_v3": models.inception_v3()

    }
    # Initialize cuda parameters
    use_cuda = torch.cuda.is_available()
    np.random.seed(2020)
    torch.manual_seed(2020)
    device = torch.device("cuda" if use_cuda else "cpu")

    print("Available device = ", device)
    model = model_collections_dict[model_name]
    model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    '''
    model.aux_logits = False
    num_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = torch.nn.Linear(num_ftrs, 40)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs,40)
    '''
    '''                                                                     squeezenet改输出特征数3
    model.classifier[1] = torch.nn.Conv2d(512, 40, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = 40
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
    # Imagnet values
    mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

    #    mean=[0.485, 0.456, 0.406]
    #    std=[0.229, 0.224, 0.225]

    root = os.getcwd() + '/data/stanford40/'  # 调用图像

    # Load the best weights before testing
    weights_file_path = os.path.join(model_dir, "model-{}.pth".format(num))

    torch.cuda.empty_cache()

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

    dataset_test = MyDataset(txt=root + 'test.txt', transform=transformations_test)

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
    main('data/', "squeezenet1_1", num=1, lr=[1.5e-4, 5e-2], epochs=10, batch_size=16, download_data=True,
         save_results=True)

