import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

assert len(sys.argv) >= 2, "out文件的路径给一下"
filename = sys.argv[1]
with open(filename, 'r') as f:
    print('Loading file.')
    lines = f.readlines()

data_columns = ['Model name', 'Epoch', 'Loss', 'Val_loss', 'Learning rate']
datas = []

model_names = []
pass_count = 0
printed = False
for idx,line in enumerate(lines):
    
    # 跳过已经确定的行
    if pass_count > 0:
        pass_count -= 1
        continue

    # 确定模型名称
    if line.startswith('Model: \"'):
        model_name = line.split('\"')[1]
        model_names.append(model_name)
        datas.append([])
        print('Reading outputs of {}.'.format(model_name))

    # 自动提取参数
    elif line.startswith('Epoch'):
        
        # Epoch 00001: LearningRateScheduler reducing learning rate to 0.0010000000474974513.
        split_pos = line.find('to')
        if split_pos != -1:
            lr_s = line[split_pos+3:-2]
            lr = eval(lr_s)
            
            # Epoch 1/1000
            s = lines[idx+1]
            epoch_s = s[6:s.find('/')]
            # if not printed:
            #     print("{}, {}".format(epoch_s, s.find('/')))
            #     printed = True
            epoch = eval(epoch_s)

            # 9/8 - 2s - loss: 0.9037 - val_loss: 1.7154
            if lines[idx+2].startswith("Model saved"):
                s = lines[idx+3]
                pass_count = 4
            else:
                s = lines[idx+2]
                pass_count = 3

            s_split = s.split('-')
            loss_s = s_split[-2][6:]
            loss = eval(loss_s)
            val_loss_s = s_split[-1][10:]
            val_loss = eval(val_loss_s)
            
            # 添加数据
            datas[-1].append([model_names[-1], epoch, loss, val_loss, lr])

dataframes = [pd.DataFrame(data=data, columns=data_columns) for data in datas]

# 绘图
print("Drawing plot.")

fig_count = len(model_names)
fig, axs = plt.subplots(nrows=1, ncols=fig_count, sharey=False, figsize=(fig_count*4, 4))

for idx, model_name in enumerate(model_names):
        i = idx
        try:
            subplot = axs[i]
        except Exception:
            subplot = axs

        epochs = dataframes[i]['Epoch']

        lrs = dataframes[i]['Learning rate']
        subplot_r = subplot.twinx()
        subplot_r.set_ylabel('Learning rate')
        lr_plot, = subplot_r.plot(epochs, lrs, 'g--')

        losses = dataframes[i]['Loss']
        loss_plot, = subplot.plot(epochs, losses)

        val_losses = dataframes[i]['Val_loss']
        val_loss_plot, = subplot.plot(epochs, val_losses)

        subplot.set_xlabel('Epoch')
        subplot.set_ylabel('Loss and val_loss')
        subplot.set_title(model_name)
        subplot.legend([loss_plot, val_loss_plot, lr_plot], data_columns[-3:])

fig.suptitle(filename)
fig.tight_layout()
fig.savefig(filename+"_anyl.png")

print("Figure saved to {}".format(filename+"_anyl.png"))

# 导出表格
data = np.vstack(datas)
dataframe = pd.DataFrame(data, columns=data_columns)
dataframe.to_csv(filename+"_anyl.csv")

print("CSV file saved to " + filename + "_anyl.csv")
