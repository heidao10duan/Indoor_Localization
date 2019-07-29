import time
import urllib.parse

from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.core.files.base import ContentFile

from DeviceManagement.models import DeviceGroup, Device, Ads
from RoomManagement.models import BuildingFloor

from .common import get_request, out_func, page_num_each
from CommonFunc.common import decode_base64


# -------------------------------------------------------

# 制作广告
def add_ads(request):
    def in_func():
        url, image, text, mac, duration = get_request(request, 'url', 'image', 'text', 'mac', 'duration')

        if duration[-1] not in ['h', 'm', 's']:
            return JsonResponse({"data": "请确认参数duration的单位是否正确"})

        # 保存图片
        data = decode_base64(image.split(',')[1])
        pic_suffix = image.split(',')[0].split('/')[1].split(';')[0]
        file_content = ContentFile(data)

        try:
            float(duration[:-1]) # 判断是否是整数或小数的时间表示
            duration = str_to_sec(duration)
            ads = Ads(ads_url=url, ads_context=text, mac=mac, ads_duration=duration)
        except:
            ads = Ads(ads_url=url, ads_context=text, mac=mac)

        ads.ads_image.save("{0}_{1}.{2}".format("广告", time.strftime("%Y%m%d%H", time.localtime()), pic_suffix),
                           file_content)
        ads.save()
        return JsonResponse({"data": "OK"})

    return out_func(request, in_func)


# 小时，分钟都转化成秒
def str_to_sec(str_time):
    unit = str_time[-1]
    coef = 1

    if unit == 'h':
        coef = 3600
    elif unit == 'm':
        coef = 60

    return int(float(str_time[:-1]) * coef)


# 获取广告
def get_ads(request):
    def in_func():
        page, = get_request(request, 'page')
        try:
            page = int(page)
        except:
            page = 1

        ads = Ads.objects.all()
        paginator = Paginator(ads, page_num_each)
        page_result = paginator.page(page)
        result = []

        for item in page_result:
            temp = {}
            temp["ads_image"] = urllib.parse.unquote(item.ads_image.url)
            temp["ads_url"] = item.ads_url
            temp["ads_context"] = item.ads_context
            temp["mac"] = item.mac
            result.append(temp)

        return JsonResponse({"data": result, "number": paginator.num_pages, "total": len(ads)})

    return out_func(request, in_func)


# 删除广告
def delete_ads(request):
    def in_func():
        ads_id, = get_request(request, 'ads_id')

        if not isinstance(ads_id, int):
            return JsonResponse({"data": "请传递正确的ads_id"})

        ads = Ads.objects.filter(id=ads_id).delete()
        if ads[0] <= 0:
            return JsonResponse({"data":"不存在该广告"})
        return JsonResponse({"data":"广告删除成功"})

    return out_func(request, in_func)


# 广告推送
def push_ads(request):
    def in_func():
        floor_id, = get_request(request, 'floor_id')
        mac_lst = list(set(Device.objects.filter(floor=floor_id).values_list('mac', flat=True)))
        ads = list(
            Ads.objects.filter(mac__in=mac_lst).values('id', 'ads_url', 'ads_image', 'ads_context', 'ads_duration'))

        return JsonResponse({"data": ads})

    return out_func(request, in_func)


# 推送广告
def push_ads1(request):
    def in_func():
        # 根据小程序发送的RSSI，计算距离最近的前三个mac
        appid, secret, js_code, floor_id, rssi_data = get_request(request, 'appid', 'secret', 'js_code', 'floor_id',
                                                                  'rssi_data')

        device = Device.objects.filter(floor=floor_id)
        device_rssi = {}

        for item in rssi_data:
            device_id = item['deviceId'].replace(':','')
            result = device.filter(mac=device_id)
            if len(list(result)) <= 0:
                continue

            RSSI = item['RSSI']
            if device_id not in device_rssi:
                device_rssi.setdefault(device_id,[])

            device_rssi[device_id].append(RSSI)

        # 求RSSI均值后排序，对前三位的device进行广告推送
        device_lst = sorted(device_rssi.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)
        top_num = 4
        if len(device_lst) < top_num:
            top_num = len(device_lst)

        mac_lst = []

        for i in range(top_num):
            mac_lst.append(device_lst[i][0])

        ads = list(Ads.objects.filter(mac__in=mac_lst).values('id', 'ads_url', 'ads_image', 'ads_context', 'ads_duration'))

        return JsonResponse({"data": ads})

    return out_func(request, in_func)