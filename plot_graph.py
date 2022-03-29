
import matplotlib.pyplot as plt
import pandas as pd

model_list =['VanillaRNN', 'BidirectionalLSTM', 'Autoencoder', 'Transformer']


for idx, model in enumerate(model_list):
    globals()['df%s' % idx] = pd.read_csv( './reports/figures/' + model + "_AUROC.csv", header=None,
    usecols=[1,2], names=['colA','colB'])

    
x0= df0['colA']
y0= df0['colB']
x1= df1['colA']
y1= df1['colB']
x2= df2['colA']
y2= df2['colB']
x3= df3['colA']
y3= df3['colB']
print(x0)
print(y0)

plt.plot(x0, y0, label = "VanillaRNN", linestyle="-")
plt.plot(x1, y1, label = "BidirectionalLSTM", linestyle="--")
plt.plot(x2, y2, label = "Autoencoder", linestyle="-.")
plt.plot(x3, y3, label = "Transformer", linestyle=":")

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Task 2 AUROC")
plt.legend()
plt.show()