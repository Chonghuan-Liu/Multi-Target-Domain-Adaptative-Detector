# #_*_coding:utf-8 _*_
# import numpy as np
# import visdom
# import time
#
# viz = visdom.Visdom(env="Test1") # 创建环境名为Test1
# #单张图像显示与更新demo
# image = viz.image(np.random.rand(3,256,256),opts={'title':'image1','caption':'How random.'})
# for i in range(10):
#     viz.image(np.random.randn( 3, 256, 256),win = image)
#     time.sleep(0.5)


# import fire
# import datetime
#
# def cal_days(date_str1, date_str2):
#     '''计算两个日期之间的天数'''
#
#     date_str1 = str(date_str1)
#     date_str2 = str(date_str2)
#     d1 = datetime.datetime.strptime(date_str1, '%Y%m%d')
#     d2 = datetime.datetime.strptime(date_str2, '%Y%m%d')
#     delta = d1 - d2
#     return delta.days
#
#
# if __name__ == '__main__':
#     fire.Fire(cal_days)

str="jjjj+ssss"
print(str.split('+'))