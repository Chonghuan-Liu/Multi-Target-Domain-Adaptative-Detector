'''
GetFolderFiles(path):
Returns [f1,f2,f3] f.dtype=str

DeleteImgXml(list):
for i 0->len(list):
    if with bbox:continue
    else:get img_file_name,delete img,delete xml
'''

import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import random
import cv2 as cv
import numpy as np
import random
#-------------------------------------------剔除没有bbox的文件
# def XmlToImg(path):
#     return str(path)[:-4]+'.jpg'
#
# filename=[]
# for _, _, filename in os.walk('./FoggyCity/Annotations/'):
#     pass
# flag=0
# for xml_path in tqdm(filename):
#     anno = ET.parse(
#         os.path.join("./FoggyCity/Annotations/", xml_path))
#     bbox = list()
#     for obj in anno.findall('object'):
#         bndbox_anno = obj.find('bndbox')
#         bbox.append([
#             int(bndbox_anno.find(tag).text) - 1
#             for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
#
#     if len(bbox)==0:
#         os.remove('./FoggyCity/JPEGImages/'+XmlToImg(xml_path))
#         os.remove('./FoggyCity/Annotations/'+xml_path)


# filename=[]
# for _, _, filename in os.walk('./FoggyCity/Annotations/'):
#     pass
# flag=0
# for xml_path in tqdm(filename):
#     print(xml_path)
#
# --------------------------------------------------制作新的训练文件的txt文件
# resultList=random.sample(range(0,13836),10000)
# trainlist=resultList[:8000]
# testlist=resultList[8000:]
# train_str=""
# test_str=""
#
# def XmlToStr(path):
#     return str(path)[:-4]
#
# filename=[]
# for _, _, filename in os.walk('./FoggyCity/Annotations/'):
#     pass
# flag=0
#
#
# for i in trainlist:
#     train_str+=(XmlToStr(filename[i])+'\n')
# for i in testlist:
#     test_str+=(XmlToStr(filename[i])+'\n')
#
# f=open('./NewMain/trainval.txt','w')
# f.write(train_str)
# f.close()
#
# f=open('./NewMain/test.txt','w')
# f.write(test_str)
# f.close()
#
#
#------------------------------------------制作有雨的图片
# def get_noise(img, value=100):
#     '''
#     #生成噪声图像
#      输入： img图像
#
#         value= 大小控制雨滴的多少
#       返回图像大小的模糊噪声图像
#     '''
#
#     noise = np.random.uniform(0, 256, img.shape[0:2])
#     # 控制噪声水平，取浮点数，只保留最大的一部分作为噪声
#     v = value * 0.01
#     noise[np.where(noise < (256 - v))] = 0
#
#     # 噪声做初次模糊
#     k = np.array([[0, 0.1, 0],
#                   [0.1, 8, 0.1],
#                   [0, 0.1, 0]])
#
#     noise = cv.filter2D(noise, -1, k)
#
#     # 可以输出噪声看看
#     # cv.imshow('img',noise)
#     # cv.waitKey()
#     # cv.destroyWindow('img')
#     return noise
#
#
# def rain_blur(noise, length=60, angle=30, w=5):
#     '''
#     将噪声加上运动模糊,模仿雨滴
#
#
#     noise：输入噪声图，shape = img.shape[0:2]
#     length: 对角矩阵大小，表示雨滴的长度
#     angle： 倾斜的角度，逆时针为正
#     w:      雨滴大小
#
#
#
#     '''
#
#     # 这里由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正
#     trans = cv.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
#     dig = np.diag(np.ones(length))  # 生成对焦矩阵
#     k = cv.warpAffine(dig, trans, (length, length))  # 生成模糊核
#     k = cv.GaussianBlur(k, (w, w), 0)  # 高斯模糊这个旋转后的对角核，使得雨有宽度
#
#     # k = k / length                         #是否归一化
#
#     blurred = cv.filter2D(noise, -1, k)  # 用刚刚得到的旋转后的核，进行滤波
#
#     # 转换到0-255区间
#     cv.normalize(blurred, blurred, 0, 255, cv.NORM_MINMAX)
#     blurred = np.array(blurred, dtype=np.uint8)
#
#     # cv.imshow('img',blurred)
#     # cv.waitKey()
#     #cv.destroyWindow('img')
#     return blurred
#
#
# def alpha_rain(rain, img, beta=0.6):
#     # 输入雨滴噪声和图像
#     # beta = 0.8   #results weight
#     # 显示下雨效果
#
#     # expand dimensin
#     # 将二维雨噪声扩张为三维单通道
#     # 并与图像合成在一起形成带有alpha通道的4通道图像
#     rain = np.expand_dims(rain, 2)
#     #print(img.shape,rain.shape)
#     rain_effect = np.concatenate((img, rain), axis=2)  # add alpha channel
#
#     rain_result = img.copy()  # 拷贝一个掩膜
#     rain = np.array(rain, dtype=np.float32)  # 数据类型变为浮点数，后面要叠加，防止数组越界要用32位
#     rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
#     rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
#     rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
#     # 对每个通道先保留雨滴噪声图对应的黑色（透明）部分，再叠加白色的雨滴噪声部分（有比例因子）
#
#     # cv.imshow('rain_effct_result', rain_result)
#     # cv.waitKey()
#     # cv.destroyAllWindows()
#     return rain_result
# #
# # ori_img=cv.imread("./Generated_img/ori/source_aachen_000001_000019_leftImg8bit.jpg")
# def generate_rain_img(ori_img,size=30,length=60,angle=30,w=5):
#     return alpha_rain(rain_blur(get_noise(ori_img,value=size),length=length,angle=angle,w=w),ori_img)
# #
# # cv.imwrite('./Generated_img/rain/0.jpg',generate_rain_img(ori_img))
#
# #----------------------------------------------------------------批量给图片加雨
#
def XmlToImg(path):
    return str(path)[:-13]

filename=[]
for _, _, filename in os.walk('./Generated_img/rain/'):
    pass

for ori_file in tqdm(filename):
    cv.imwrite('D:\\rain_image\\'+str(XmlToImg(ori_file))+'.jpg',
               cv.imread('./Generated_img/rain/'+ori_file))
# for ori_img_name in tqdm(filename):
#     size=random.randint(30,90)
#     angle=random.randint(10,45)
#     cv.imwrite('./Generated_img/rain/'+str(XmlToImg(ori_img_name))+'_rain.jpg',
#                generate_rain_img(cv.imread('./FoggyCity/JPEGImages/'+str(ori_img_name)),size=size,angle=angle))









































































