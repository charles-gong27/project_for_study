import torch
from torchvision import transforms
import torchvision.models as  models
from torch.utils.data import DataLoader
import os
from dataset import MyDataset
import numpy as np
from utils import get_ap_score,dealit
from tqdm import tqdm
olderr = np.seterr(all='ignore')
#classes = ('jumping', 'phoning', 'playinginstrument', 'reading', 'ridingbike', 'ridinghorse', 'running', 'takingphoto', 'usingcomputer', 'walking','others')
#'''
classes = ('applauding', 'blowing_bubbles', 'brushing_teeth',
            'cleaning_the_floor', 'climbing', 'cooking', 'cutting_trees',
            'cutting_vegetables', 'drinking', 'feeding_a_horse',
            'fishing', 'fixing_a_bike', 'fixing_a_car', 'gardening',
            'holding_an_umbrella', 'jumping', 'looking_through_a_microscope',
            'looking_through_a_telescope', 'playing_guitar', 'playing_violin',
            'pouring_liquid', 'pushing_a_cart', 'reading', 'phoning',
            'riding_a_bike', 'riding_a_horse', 'rowing_a_boat', 'running',
            'shooting_an_arrow', 'smoking', 'taking_photos', 'texting_message',
            'throwing_frisby', 'using_a_computer', 'walking_the_dog',
            'washing_dishes', 'watching_TV', 'waving_hands', 'writing_on_a_board', 'writing_on_a_book')
#'''
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
    #import cbowkai
    #from cbowkai import modelll, make_context_vector, idx_to_word, word_to_idx, nomprebab

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
            output = model(data.view(-1, c, h, w))

            output = output.view(bs, ncrops, -1).mean(1)

            loss = criterion(output, target)

            running_loss += loss  # sum up batch loss
            running_ap += get_ap_score(torch.Tensor.cpu(target).detach().numpy(),
                                       torch.Tensor.cpu(m(output)).detach().numpy())

            #print(m(output),target)

            if returnAllScores == True:
                all_scores = np.append(all_scores, torch.Tensor.cpu(m(output)).detach().numpy(), axis=0)
                ground_scores = np.append(ground_scores, torch.Tensor.cpu(target).detach().numpy(), axis=0)

            del data, target, output
            torch.cuda.empty_cache()

    num_samples = float(len(test_loader.dataset))
    avg_test_loss = running_loss.item() / num_samples
    test_map = running_ap / num_samples

    print('test_loss: {:.4f}, test_avg_precision:{:.3f}'.format(
        avg_test_loss, test_map))
    return avg_test_loss,test_map




def main(data_dir, model_name, num, lr, epochs, batch_size=16, download_data=False, save_results=False):

    model_dir = os.path.join("./models", model_name)
    #model_dir='./models/'+model_name
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    print('------start---------')
    model_collections_dict = {
        "resnet34": models.resnet34(),
        #"densenet121": models.densenet121(),

    }
    # Initialize cuda parameters
    use_cuda = torch.cuda.is_available()
    np.random.seed(2020)
    torch.manual_seed(2020)
    device = torch.device("cuda" if use_cuda else "cpu")

    print("Available device = ", device)
    model = model_collections_dict[model_name]
    model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    '''                                                                     densenet改输出特征数
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_ftrs, 40)
    '''
    #'''                                                                     resnet改输出特征数
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 40)                                                                                        #数据集时需改动
    #'''
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

    for qwe in classes:
        dataset_test = MyDataset(txt=root + qwe + '_testfortype.txt', transform=transformations_test)
        test_loader = DataLoader(dataset_test, batch_size=3, num_workers=0, shuffle=False)  # int(batch_size / 5)
        losss, mappp = test(model, device, test_loader, returnAllScores=True)
        f = open((qwe+'_test_result.txt'), 'w')
        f.write('test loss:' + str(losss) + '          ' + 'test_ap:' + str(mappp))
        f.close()


# Execute main function here
if __name__ == '__main__':
    main('data/', "resnet34", num=1, lr=[1.5e-4, 5e-2], epochs=10, batch_size=16, download_data=True,
         save_results=True)

