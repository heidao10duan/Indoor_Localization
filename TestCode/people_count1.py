# -*- coding: utf-8 -*-
# !/usr/bin/env python
import os
import requests
import base64
import json
from pprint import pprint
import time
import io
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
import glob
from aip import AipBodyAnalysis

APP_ID = "16855128"
API_KEY = "7Qurlx6Kz79yICgqFO9Ab2p7"
SECRET_KEY = "EwpDfYtW7dzDOmky5wZnOYaz678U1n0Z"
client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)

# client_id 为官网获取的AK， client_secret 为官网获取的SK
api_key = '16859423'
secret_key = 'gswQ9ZjQqF4GloSEc7Iq6K5GD3MRpoGy'


class Traffic_flowRecognizer(object):
    def __init__(self, api_key, secret_key):
        # self.access_token = self._get_access_token(api_key=api_key, secret_key=secret_key)
        self.access_token = "24.f67749bf79dc9d63c9da783d8802cc8d.2592000.1566353819.282335-16855128"
        self.API_URL = 'https://aip.baidubce.com/rest/2.0/image-classify/v1/body_tracking' + '?access_token=' \
                       + self.access_token
        # 获取token

    @staticmethod
    def _get_access_token(api_key, secret_key):
        api = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials' \
              '&client_id={}&client_secret={}'.format(api_key, secret_key)
        rp = requests.post(api)
        if rp.ok:
            rp_json = rp.json()
            # print(rp_json['access_token'])
            return rp_json['access_token']
        else:
            print('=>Error in get access token!')

    def get_result(self, params):
        rp = requests.post(self.API_URL, data=params)
        if rp.ok:
            # print('=>Success! got result: ')
            rp_json = rp.json()
            # print(rp_json)
            return rp_json
        else:
            # print('=>Error! token invalid or network error!')
            # print(rp.content)
            return None
            # 人流量统计

    def detect(self):
        ###对视频进行抽帧后，抽帧频率5fps，连续读取图片
        WSI_MASK_PATH = 'C:/Users/Juan/Desktop/11/'  # 存放图片的文件夹路径

        paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.jpg'))
        print (paths[0])
        paths.sort(key=lambda a:int(a[36:-4]))
        data_list = []
        c = 1
        count = 0
        for path in paths:
            # if count < 20:
            #     count +=1
            #     continue
            # else:
            #     count = 0
            # print ("*************")
            # print (path)
            # continue
            c = c + 1
            f = open(path, 'rb')
            image = f.read()
            # client.bodyNum(image)
            img_str = base64.b64encode(image)
            data_list.append(img_str)
            if c % 20 != 0:
                continue


            params = {'dynamic': 'true', 'area': "100,100,419,100,419,419,100,419", 'case_id': 1213, 'case_init': 'false',
                      'image': data_list, 'show': 'true'}

            tic = time.clock()
            rp_json = self.get_result(params)
            toc = time.clock()
            print('单次处理时长: ' + '%.2f' % (toc - tic) + ' s')
            # print ("**___________")
            print ("person_num:",rp_json["person_num"])
            print ("person_count:",rp_json["person_count"])
            if "image" not in rp_json:
                continue
            img_b64encode = rp_json['image']
            img_b64decode = base64.b64decode(img_b64encode)  # base64解码
            # 显示检测结果图片
            #            image = io.BytesIO(img_b64decode)
            #            img = Image.open(image)
            #            img.show()
            # 存储检测结果图片
            file = open('C:/Users/Juan/Desktop/1/' + str(c) + '.jpg', 'wb')
            # file = open('E:/renliu/out/' + str(c) + '.jpg', 'wb')
            file.write(img_b64decode)
            file.close()
            # c = c + 1


if __name__ == '__main__':
    recognizer = Traffic_flowRecognizer(api_key, secret_key)
    recognizer.detect()