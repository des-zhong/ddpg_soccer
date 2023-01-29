# DDPG
DDPG.py     DDPG结构，设置noise为0.05
networks.py  Actor和Critic网络结构，其中action最后一层采用了tanh激活函数限制范围(-1，1)
soccertrain.py  训练脚本，每个球员用一个独立的网络
soccertrainv1.py  训练脚本，进攻方2个球员共用一个网络，防守方随机运动
奖励函数在utility的runstep中
在control.py中可以修改网络参数的读取路径，修改后运行main.py可以看到当前网络参数的效果

#调用接口
在Offense文件夹下面储存了训练的参数，调用control.py会根据当前的state返回待采取的command，由于目前仿真中单位为mm，所以control输出的command为action*100。
