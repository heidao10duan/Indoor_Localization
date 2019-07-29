import cv2
from PIL import Image
import base64
import urllib
import urllib.request
import urllib.parse
import json
name = 'renliuliang'


def getface(path):
  # cap = cv2.VideoCapture("http://hls01open.ys7.com/openlive/9006facdc13a4611a36700fd7486515a.hd.m3u8")
  cap = cv2.VideoCapture(path)
  print ("**cap:",cap)
  class_path = "C:/Software/Anaconda3/envs/tensorflow/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml"
  classfier = cv2.CascadeClassifier(class_path)

  suc = cap.isOpened()  # 是否成功打开
  frame_count = 0
  out_count = 0
  count = 0

  width = 640
  height = 480

  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter('out.avi', fourcc, 20.0, (width, height))
  count = 0
  while cap.isOpened():
      ret, frame = cap.read()
      if ret is True:
          frame = cv2.resize(frame, (640, 480))

          out.write(frame)

          cv2.imshow('frame', frame)

          # if count < 10:
          #     count +=1
          #     continue
          # else:
          #     count = 0
          params = []
          params.append(2)
          image_path = 'C:/Users/Juan/Desktop/11/' + name + '%d.jpg' % out_count
          cv2.imwrite(image_path, frame, params)  # 存储到指定目录
          out_count += 1
          continue

          '''
                            人流量统计（动态版）
                            '''

          request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/body_tracking"

          # 二进制方式打开图片文件
          f = open(image_path, 'rb')
          img = base64.b64encode(f.read())

          params = {"area": "1,1,639,1,639,479,1,479", "case_id": 1, "case_init": "false", "dynamic": "true",
                    "image": img}
          params = urllib.parse.urlencode(params).encode(encoding='UTF8')

          access_token = '24.f67749bf79dc9d63c9da783d8802cc8d.2592000.1566353819.282335-16855128'
          request_url = request_url + "?access_token=" + access_token
          request = urllib.request.Request(url=request_url, data=params)
          request.add_header('Content-Type', 'application/x-www-form-urlencoded')
          response = urllib.request.urlopen(request)
          content = response.read()
          if content:
              print(content)

          params = []
          params.append(2)  # params.append(1)
          grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前桢图像转换成灰度图像
          faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))  # 读取脸部位置
          if len(faceRects) > 0:  # 大于0则检测到人脸
              for faceRect in faceRects:  # 单独框出每一张人脸
                  x, y, w, h = faceRect
                  # image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                  image = frame
                  # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #转为灰度图
                  # img_new = cv2.resize(image,(47,57), interpolation = cv2.INTER_CUBIC) #处理面部的大小
                  image_path = 'C:/Users/Juan/Desktop/11/' + name + '%d.jpg' % out_count
                  cv2.imwrite(image_path, image, params)  # 存储到指定目录
                  out_count += 1
                  print('成功提取' + name + '的第%d个脸部' % out_count)



                  break  # 每帧只获取一张脸，删除这个即为读出全部面部


      else:
          break

      key = cv2.waitKey(1)
      if key == ord('q'):
          break
  #
  # while suc:
  #     frame_count += 1
  #     if out_count > 599: #最多取出多少张
  #       break
  #     suc, frame = cap.read() #读取一帧
  #     if not suc:
  #         continue
  #     frame = cv2.resize(frame, (640, 480))
  #     out.write(frame)
  #     # if count <=5:
  #     #     count += 1
  #     #     continue
  #     # else:
  #     #     count = 0
  #     cv2.imshow('frame', frame)
  #     params = []
  #     params.append(2)  # params.append(1)
  #     grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #将当前桢图像转换成灰度图像
  #     faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32)) #读取脸部位置
  #     if len(faceRects) > 0:          #大于0则检测到人脸
  #           for faceRect in faceRects:  #单独框出每一张人脸
  #               x, y, w, h = faceRect
  #               image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
  #               # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #转为灰度图
  #               # img_new = cv2.resize(image,(47,57), interpolation = cv2.INTER_CUBIC) #处理面部的大小
  #               cv2.imwrite('C:/Users/Juan/Desktop/11/'+ name +'%d.jpg' % out_count, image, params) #存储到指定目录
  #               out_count += 1
  #               print('成功提取'+ name +'的第%d个脸部'%out_count)
  #               break #每帧只获取一张脸，删除这个即为读出全部面部
  # cap.release()
  # cv2.destroyAllWindows()
  # print('总帧数:', frame_count)
  # print('提取脸部:',out_count)


def test():
    # 导入cv模块
    import cv2 as cv

    # 读取图像，支持 bmp、jpg、png、tiff 等常用格式
    img = cv.imread("C:/Users/Juan/Desktop/11.jpg")
    # "D:\Test\2.jpg"
    # 创建窗口并显示图像
    cv.namedWindow("Image")
    cv.imshow("Image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

import numpy as np
import os


def remove_simillar_picture_by_perception_hash(path):
    img_list = os.listdir(path)
    hash_dic = {}
    hash_list = []
    count_num = 0
    for img_name in img_list:
        try:
            img = cv2.imread(os.path.join(path, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            count_num += 1
            print(count_num)
        except:
            continue

        img = cv2.resize(img, (8, 8))

        avg_np = np.mean(img)
        img = np.where(img > avg_np, 1, 0)
        hash_dic[img_name] = img
        if len(hash_list) < 1:
            hash_list.append(img)
        else:
            for i in hash_list:
                flag = True
                dis = np.bitwise_xor(i, img)

                if np.sum(dis) < 22:
                    flag = False
                    os.remove(os.path.join(path, img_name))
                    break
            if flag:
                hash_list.append(img)


# def remove_simillar_image_by_ssim(path):
#     img_list = os.listdir(path)
#     img_list.sort()
#     hash_dic = {}
#     save_list = []
#     count_num = 0
#     for i in range(len(img_list)):
#         try:
#             img = cv2.imread(os.path.join(path, img_list[i]))
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             img = cv2.resize(img, (256, 256))
#             count_num += 1
#         except:
#             continue
#         if count_num == 1:
#             save_list.append(img_list[i])
#             continue
#         elif len(save_list) < 5:
#             flag = True
#             for j in range(len(save_list)):
#                 com_img = cv2.imread(os.path.join(path, save_list[j]))
#                 com_img = cv2.cvtColor(com_img, cv2.COLOR_BGR2GRAY)
#                 com_img = cv2.resize(com_img, (256, 256))
#                 sim = compare_ssim(img, com_img)
#                 if sim > 0.4:
#                     os.remove(os.path.join(path, img_list[i]))
#                     flag = False
#                     break
#             if flag:
#                 save_list.append(img_list[i])
#         else:
#             for save_img in save_list[-5:]:
#                 com_img = cv2.imread(os.path.join(path, save_img))
#                 com_img = cv2.cvtColor(com_img, cv2.COLOR_BGR2GRAY)
#                 com_img = cv2.resize(com_img, (256, 256))
#                 sim = compare_ssim(img, com_img)
#                 if sim > 0.4:
#                     os.remove(os.path.join(path, img_list[i]))
#                     flag = False
#                     break
#             if flag:
#                 save_list.append(img_list[i])


import sys

def main_1():
    print ("***********************")
    print (sys._getframe().f_code.co_name)
    return "Hello World"


def test11(floor_id):
    print (eval("main_{0}".format(floor_id))())


def people_count(path):
    lst = []

    for item in os.listdir(path):
        '''
        人流量统计（动态版）
        '''
        image_path = os.path.join(path, item)

        request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/body_tracking"

        # 二进制方式打开图片文件
        f = open(image_path, 'rb')
        img = base64.b64encode(f.read())

        params = {"area": "1,1,639,1,639,279,1,279", "case_id": 1, "case_init": "false", "dynamic": "true","show":"true",
                  "image": img}
        params = urllib.parse.urlencode(params).encode(encoding='UTF8')

        access_token = '24.f67749bf79dc9d63c9da783d8802cc8d.2592000.1566353819.282335-16855128'
        request_url = request_url + "?access_token=" + access_token
        request = urllib.request.Request(url=request_url, data=params)
        request.add_header('Content-Type', 'application/x-www-form-urlencoded')
        response = urllib.request.urlopen(request)
        content = response.read()

        if content:
            content = json.loads(content)
            for item in content["person_info"]:
                print (item)
                print (item["ID"])
                lst.append(item["ID"])
            print (content)

    print ("***********lst")
    print (lst)
    print (set(lst))


if __name__ == '__main__':
    getface("C:/Users/Juan/Desktop/"+ name +".mp4") #参数为视频地址
    # test()
    # remove_simillar_picture_by_perception_hash("C:/Users/Juan/Desktop/11")

    # people_count(r'C:\Users\Juan\Desktop\11')