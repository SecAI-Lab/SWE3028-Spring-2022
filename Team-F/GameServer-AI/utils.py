from sklearn.model_selection import train_test_split
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

def Preprocess_Dataset(x_data,y_data):
    # 28x28
    x_data=x_data.reshape(x_data.shape[0],28,28)
    # 32x32
    x_data=np.array([np.pad(x,((2,2),(2,2)),'constant',constant_values=0) for x in x_data])
    x_data=x_data.astype('float32')/255.0
    y_data=y_data.astype('int64')

    x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,stratify=y_data,test_size=0.1)
    x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,stratify=y_train,test_size=0.1)

    print(f'>>> train data => x_data : {x_train.shape} | y_data : {y_train.shape}' )
    print(f'>>>   val data => x_data :  {x_val.shape} | y_data : {y_val.shape}' )
    print(f'>>>  test data => x_data : {x_test.shape} | y_data : {y_test.shape}' )

    return (x_train,y_train),(x_val,y_val),(x_test,y_test)


class CustomDataset(Dataset):
    def __init__(self,x_data,y_data,transform):
        self.x_data=x_data
        self.y_data=y_data
        self.len=self.y_data.shape[0]
        self.transform=transform

    def __getitem__(self,index):
        x=self.transform(self.x_data[index])
        y=self.y_data[index]
        return x,y

    def __len__(self):
        return self.len


def Dataset_with_Augmentation(train_set,val_set,test_set):

    x_train,y_train=train_set
    x_val,y_val=val_set
    x_test,y_test=test_set

    train_dataset=CustomDataset(x_train,y_train,transforms.Compose([transforms.ToTensor(),
                                                                    transforms.RandomCrop(24),
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.RandomRotation(20),
                                                                    transforms.Resize(32),]))

    val_dataset=CustomDataset(x_val,y_val,transforms.Compose([transforms.ToTensor(),
                                                              transforms.Resize(32)]))

    test_dataset=CustomDataset(x_test,y_test,transforms.Compose([transforms.ToTensor(),
                                                                transforms.RandomCrop(24),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.RandomRotation(20),
                                                                transforms.Resize(32)]))


    return train_dataset,val_dataset,test_dataset


def Dataset_with_No_Augmentation(train_set,val_set,test_set):

    x_train,y_train=train_set
    
    x_val,y_val=val_set
    
    x_test,y_test=test_set
    
    train_dataset=CustomDataset(x_train,y_train,transforms.Compose([transforms.ToTensor(),transforms.Resize(32)]))

    val_dataset=CustomDataset(x_val,y_val,transforms.Compose([transforms.ToTensor(),transforms.Resize(32)]))

    test_dataset=CustomDataset(x_test,y_test,transforms.Compose([transforms.ToTensor(),transforms.Resize(32)]))

    return train_dataset,val_dataset,test_dataset


def get_DataLoader(train_dataset,val_dataset,test_dataset,batch_size=64):

    train_loader=DataLoader(train_dataset,batch_size,shuffle=True,drop_last=True)

    val_loader=DataLoader(val_dataset,batch_size,shuffle=True,drop_last=True)

    test_loader=DataLoader(test_dataset,batch_size,shuffle=False,drop_last=False)

    return train_loader,val_loader,test_loader



