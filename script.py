import argparse
import os
import torch
import torchvision
import transforms as T
import utils

from azureml.core import Dataset, Run
from data import PennFudanDataset
from engine import train_one_epoch, evaluate
from model import get_instance_segmentation_model

NUM_CLASSES = 2


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    print("Torch version:", torch.__version__)
    # get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="penn_ds",
                        help='name of dataset for training and test')
    parser.add_argument('--model_name', type=str, default="pytorch-peds",
                        help='name with which to register your model')
    parser.add_argument('--output_dir', default="outputs",
                        type=str, help='output directory')
    parser.add_argument('--n_epochs', type=int,
                        default=10, help='number of epochs')
    args = parser.parse_args()
    dataset_name = args.dataset_name
    run = Run.get_context()
    ws = run.experiment.workspace

    # Get a dataset by name
    penn_ds = Dataset.get_by_name(workspace=ws, name=dataset_name)

    # use our dataset and defined transformations
    dataset = PennFudanDataset(penn_ds, get_transform(train=True))
    dataset_test = PennFudanDataset(penn_ds, get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    if torch.cuda.is_available():
        print('Using GPU')
        device = torch.device('cuda')
    else:
        print('Using CPU')
        device = torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = NUM_CLASSES

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(args.n_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    # Saving the state dict is recommended method, per
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pt'))
    model = run.register_model(
        model_name='pytorch-peds',
        model_path=os.path.join(args.output_dir, 'model.pt'))
    print(model.name, model.id, model.version, sep='\t')


if __name__ == '__main__':
    main()
