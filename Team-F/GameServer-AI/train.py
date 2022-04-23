import torch
import torch.nn as nn
import time
import sys

def train(model,train_loader=None,val_loader=None,epochs=50,learning_rate=1e-4,train_on_gpu=True,path=None):
  train_loader=train_loader
  val_loader=val_loader
  
  if train_loader==None or val_loader==None:
    sys.exit('err: No loader')
  
  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f'device:{device}')

  model=model.to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  criterion = nn.CrossEntropyLoss().cuda()

  train_accuracy=[]
  val_accuracy=[]
  train_losses=[]
  val_losses=[]

  min_val_loss=-1

  #train
  for epoch in range(epochs):
    train_loss=0.0
    val_loss=0.0

    train_acc=0.0
    val_acc=0.0

    model.train()
    start = time.time()

    for i, data in enumerate(train_loader):
      x,y=data
      if train_on_gpu:
        x,y=x.to(device),y.to(device)

      optimizer.zero_grad()

      y_pred=model(x)

      loss=criterion(y_pred,y)
      loss.backward()

      optimizer.step()

      train_loss+=loss.item()*x.size(0)

      _, pred = torch.max(y_pred, dim=1)
      correct = pred.eq(y.data.view_as(pred))
      acc = torch.mean(correct.type(torch.FloatTensor))
      train_acc += acc.item() * x.size(0)

    with torch.no_grad():
      model.eval()

      for data in val_loader:
        x,y=data
        if train_on_gpu:
          x,y=x.to(device),y.to(device)
          
        y_pred=model(x)

        loss=criterion(y_pred,y)

        val_loss+=loss.item()*x.size(0)

        _,pred=torch.max(y_pred,dim=1)
        correct=pred.eq(y.data.view_as(pred))
        acc=torch.mean(correct.type(torch.FloatTensor))
        val_acc+=acc.item()*x.size(0)
        
    train_loss=train_loss/len(train_loader.dataset)
    train_losses.append(train_loss)
    val_loss=val_loss/len(val_loader.dataset)
    val_losses.append(val_loss)

    train_acc=train_acc/len(train_loader.dataset)
    train_accuracy.append(train_acc)
    val_acc=val_acc/len(val_loader.dataset)
    val_accuracy.append(val_acc)

    end = time.time()
    
    if min_val_loss==-1 or min_val_loss>val_loss:
      min_val_loss=val_loss
      torch.save(model.state_dict(),path+'/exp.pt')
    
    print(f'Epoch: {epoch}/{epochs}')
    print(f'\tTraining Loss:{train_loss:.4f} | Validation Loss:{val_loss:.4f} | Training Acc:{100*train_acc:.4f}% | Validation Acc:{100*val_acc:.4f}% \tTime :{(end-start)/60:.1f}min')
  
  history={'loss':train_losses,'val_loss':val_losses,'acc':train_accuracy,'val_acc':val_accuracy}
  return history


