import torch

def test(model,test_loader=None,total_data=None,pre_trained=True,path=None):
  if not pre_trained:
    model.load_state_dict(torch.load(path))

  total_correct=0
  total_data=total_data

  model.eval()
  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model=model.to(device)

  for data in test_loader:
    x,y=data 

    x=x.to(device)
    y=y.to(device)

    y_pred=model(x)

    _,pred=torch.max(y_pred,dim=1)
    correct=torch.sum((y==pred).int())
    total_correct+=correct/len(y)

  accuracy=total_correct*100/total_data

  print(f'accuaracy:{total_correct*100/total_data:.4f}')

  return accuracy
