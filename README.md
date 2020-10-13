# naic
文件目录

--baseline_tensorflow

--data

--model_zoo(可忽略)

data下载链接：https://awscdn.datafountain.cn/cometition_data2/Files/PengCheng2020/CSI/Hdata.mat
下载后放在data文件里即可

训练时需要修改baseline_tensorflow/Model_train.py内的文件读取和保存路径，如下：


line35 --> data_load_address = '/content/drive/My Drive/NAIC/data'  # 此处改为实际文件路径 “实际路径 + /data"

line51 --> modelsave1 = '/content/drive/My Drive/NAIC/Modelsave/encoder.h5'  # 此处改为 “实际路径 + /Modelsave/encoder.h5”

line55 --> modelsave2 = '/content/drive/My Drive/NAIC/Modelsave/decoder.h5'  # 此处改为 “实际路径 + /Modelsave/decoder.h5”

训练结束后，导出文件Modelsave。

祝你一切顺利！

