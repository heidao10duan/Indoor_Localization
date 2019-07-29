# encoding:utf-8
import base64
import urllib
import urllib.request
import urllib.parse
# import urllib2

'''
人流量统计（动态版）
'''

request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/body_tracking"

# 二进制方式打开图片文件
f = open(r'C:\Users\Juan\Desktop\mptest.mp4', 'rb')
img = base64.b64encode(f.read())


params = {"area":"1,1,1013,1,1013,772,1,772","case_id":1,"case_init":"false","dynamic":"true","image":img}
params = urllib.parse.urlencode(params).encode(encoding='UTF8')


access_token = '24.f67749bf79dc9d63c9da783d8802cc8d.2592000.1566353819.282335-16855128'
request_url = request_url + "?access_token=" + access_token
request = urllib.request.Request(url=request_url, data=params)
request.add_header('Content-Type', 'application/x-www-form-urlencoded')
response = urllib.request.urlopen(request)
content = response.read()
if content:
    print (content)
