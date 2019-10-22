# tf2.0-FCN

代码为适合入门的tensorflow2.0版本fcn32s。用tensorflow2.0的accury计算的精确率能达到91%。

拉下代码后，需要根据config新建目录,放置训练集与测试集；

训练所用的数据集为kitti，下载连接为https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip

在训练过程中会自动下载vgg模型与resnet模型，下载速度有点慢，可以在这里下载（链接：https://pan.baidu.com/s/1Pc42p404uViizGYRUpHSoQ&shfl=sharepset 
提取码：l443）

linux用户放置在/home/dennis/.keras/model/目录下即可，若没有这个目录，请先运行代码，到下载界面后再查看是否已经生成了该目录。


