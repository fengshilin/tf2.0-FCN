# tf2.0-FCN

代码为适合入门的tensorflow2.0版本fcn32s。用tensorflow2.0的accury计算的精确率能达到91%。

FCN网络结构
![](https://img-blog.csdnimg.cn/20191021141059420.png)

VGG16网络结构
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019102418154872.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzE2MjI0MA==,size_16,color_FFFFFF,t_70)

拉下代码后，需要根据config的路径,放置训练集与测试集；

训练所用的数据集为kitti，下载连接为https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip

如果下载慢，可以去这里获取网盘连接：https://blog.csdn.net/weixin_43162240/article/details/102659646

在训练过程中会自动下载vgg模型与resnet模型，下载速度有点慢，可以在这里下载（链接：https://pan.baidu.com/s/1Pc42p404uViizGYRUpHSoQ&shfl=sharepset 
提取码：l443）

linux用户放置在/home/dennis/.keras/model/目录下即可，若没有这个目录，请先运行代码，到下载界面后再查看是否已经生成了该目录。

注意：

1.代码是在linux系统下跑的，若是在windows系统下，可能需要修改其中的路径，（/改为\\）。

2.框架中的模型部分可以自己修改（只要是语义分割就行），自己定义模型。

