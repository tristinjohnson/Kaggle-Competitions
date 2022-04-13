import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm
import os
from PIL import Image
import io


# define class to read all images from S3 buckets
class S3ImageSet(Dataset):
    def __init__(self, urls, transform=None):
        super().__init__(urls)
        self.transform = transform
        self.urls = urls

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, item):
        # get images and convert from bytes to RBG
        img_name, img = super(S3ImageSet, self).__getitem__(item)
        image = Image.open(io.BytesIO(img)).convert('RBG')

        # call image transformation
        if self.transform is not None:
            image = self.transform(image)

        return image


# define model parameters
batch_size = 32
epochs = 20
learning_rate = 0.0001
num_classes = 6

# use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)


# function to apply image augmentation to training and testing images
def image_augmentation(train_dir, test_dir, data_type):
    # augmentation on training data
    train_transform = torchvision.transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205))
    ])

    # augmentation on testing data
    test_transform = torchvision.transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205))
    ])

    # augment all training and testing images
    if data_type == 's3_data':
        train_url = "s3://intel-image-classification/seg_train/"
        test_url = "s3://intel-image-classification/seg_test/"
        train_image_aug = S3ImageSet(urls=train_url, transform=train_transform)
        test_image_aug = S3ImageSet(urls=test_url, transform=test_transform)

    else:
        train_image_aug = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transform)
        test_image_aug = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform)

    # import augmented images into torch DataLoader
    train_dataloader = DataLoader(train_image_aug, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_image_aug, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


# function to get either pretrained model or custom model architecture and hyperparameters
def get_model(pretrained=True):

    # used pretrained model
    if pretrained == True:
        model = torchvision.models.resnet50(pretrained=True)
        #model = torchvision.models.googlenet(pretrained=True)
        #model = torchvision.models.vgg19(pretrained=True)
        #model = torchvision.models.efficientnet_b7(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    # set fully connected layer to Linear layer to number of classes (6)
    if model._get_name() == 'EfficientNet' or model._get_name() == 'VGG':
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.to(device)

    # define criterion, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    return model, criterion, optimizer, scheduler


# train and test the model
def train_and_test(train_ds, test_ds):

    # get model, criterion, optimizer, scheduler
    model, criterion, optimizer, scheduler = get_model(pretrained=True)
    model_best_acc = 0

    for epoch in range(1, epochs+1):
        train_loss, train_steps = 0, 0

        # put model in training mode
        model.train()

        with tqdm(total=len(train_ds), desc=f'TRAINING: Epoch {epoch}') as pbar:

            for data, target in train_ds:

                # send data and target to device
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_steps += 1

                total_loss = train_loss / train_steps
                pbar.update(1)
                pbar.set_postfix_str(f'Loss: {total_loss:0.5f}')

        # put model in evaluation mode
        model.eval()

        num_correct, total_images = 0, 0

        with torch.no_grad():
            with tqdm(total=len(test_ds), desc=f'TESTING: Epoch {epoch}') as pbar:
                for data, target in test_ds:
                    for i in range(len(target)):

                        # send data and target to device
                        data, target = data.to(device), target.to(device)
                        img = data[i].view(1, 3, 150, 150)
                        output = model(img)
                        ps = torch.exp(output)
                        pred = list(ps.cpu()[0])

                        # get pred and true label
                        pred_label = pred.index(max(pred))
                        true_label = target.cpu()[i]

                        # if true, increment num correct
                        if true_label == pred_label:
                            num_correct += 1

                        total_images += 1

                    val_accuracy = (num_correct / total_images) * 100
                    pbar.update(1)
                    pbar.set_postfix_str(f'Acc: {val_accuracy:0.5f}')

        # output final epoch metrics
        print(f'Epoch {epoch} final stats: Loss --> {total_loss:0.5f} Accuracy --> {val_accuracy:0.5f}%\n')

        model_acc = val_accuracy

        # if model accuracy is better than previous accuracy, save new model
        if model_acc > model_best_acc:
            torch.save(model.state_dict(), 'best_intel_image_model.pt')

            if epoch == 1:
                print('This model has been saved!\n')
            else:
                print('This model out-performed previous models and has been saved!\n')

            model_best_acc = model_acc

    print('Training and Testing is complete!')


if __name__ == '__main__':

    # get train, test, and pred image directories
    data_train = '../data/intel-image-classification/seg_train/seg_train'
    data_test = '../data/intel-image-classification/seg_test/seg_test'
    data_pred = '../data/intel-image-classification/seg_pred/seg_pred'

    # output total number of classes
    classes = os.listdir(data_train)
    print('Classes: ', classes)

    # perform image augmentation and get train/test dataloaders
    train_ds, test_ds = image_augmentation(data_train, data_test, data_type='')

    # train and test the model
    train_and_test(train_ds, test_ds)

