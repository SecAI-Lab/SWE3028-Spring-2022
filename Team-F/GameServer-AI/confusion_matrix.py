import matplotlib.pyplot as plt
import pandas as pd
import math
import torch

def confusion_matrix_plot(classes,test_loader):

    y_pred = []
    y_true = []
    classes=classes
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # iterate over test data
    for x, y in test_loader:
        x,y=x.to(device),y.to(device)

        y_ = model(x)

    y_ = (torch.max(torch.exp(y_), 1)[1]).data.cpu().numpy()
    y_pred.extend(y_) # Save Prediction

    y = y.data.cpu().numpy()
    y_true.extend(y) # Save Truth

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],columns = [i for i in classes])
    plt.figure(figsize = (20,15))
    sn.heatmap(df_cm, annot=True)
