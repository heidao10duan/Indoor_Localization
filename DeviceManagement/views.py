import json

from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

from DeviceManagement.models import DeviceGroup, Device
from RoomManagement.models import BuildingFloor

from .common import get_request, out_func

# Create your views here.

page_num_each = 7


# # 请求json中字段分解
# def get_request(request, *args):
#     if type(request) == dict:
#         req = request
#     else:
#         req = json.loads(request.body.decode())
#         if type(req) == str:
#             req = json.loads(req)
#     result = []
#     for item in args:
#         if item not in req:
#             result.append("")
#         else:
#             result.append(req[item])
#     return result
#
#
# def out_func(request, func):
#     if request.method != 'POST':
#         response = JsonResponse({"data": "请使用POST方法调用该接口"})
#         response.status_code = 405
#         return response
#     try:
#         return func()
#     except Exception as e:
#         response = JsonResponse({"data": str(e)})
#         response.status_code = 500
#         return response


# 添加设备分组
def add_group(request):
    def in_func():
        group_name, = get_request(request, 'name')
        device_group_count = DeviceGroup.objects.filter(name=group_name).count()
        if device_group_count <= 0:
            device_group = DeviceGroup(name=group_name)
            device_group.save()
            return JsonResponse({"data": "OK"})
        else:
            return JsonResponse({"data": "已存在该分组"})

    return out_func(request, in_func)


# 获取设备分组
def get_group(request):
    def in_func():
        page, = get_request(request, 'page')
        try:
            page = int(page)
        except:
            page = 1
        device_group = DeviceGroup.objects.values('id','name')
        paginator = Paginator(device_group, page_num_each)
        result = paginator.page(page)
        return JsonResponse({"data": list(result), "number": paginator.num_pages,"total":len(device_group)})

    return out_func(request, in_func)


def delete_group(request):
    def in_func():
        name, = get_request(request, 'name')
        device_group = DeviceGroup.objects.filter(name=name).delete()
        if device_group[0] <= 0:
            return JsonResponse({"data":"不存在该设备分组"})
        return JsonResponse({"data":"设备分组删除成功"})

    return out_func(request, in_func)


# 设备分组下拉框
def select_group(request):
    def in_func():
        device_group = list(DeviceGroup.objects.values('id', 'name'))
        return JsonResponse({"data":device_group})

    return out_func(request, in_func)


# 添加标签
def add_device(request):
    def in_func():
        mac, name, floor_id, group_id, device_type = get_request(request, 'mac', 'name', 'floor_id', 'group_id', 'device_type')
        device_group = list(DeviceGroup.objects.filter(id=group_id))

        if len(device_group) <= 0:
            return JsonResponse({"data":"不存在该分组"})

        building_floor = list(BuildingFloor.objects.filter(id=floor_id))

        if len(building_floor) <= 0:
            return JsonResponse({"data":"不存在该楼层"})

        device_count = Device.objects.filter(mac=mac).count()
        if device_count <= 0:
            device = Device(mac=mac, name=name, floor=BuildingFloor(id=floor_id), group=group_id, type=device_type)
            device.save()
            return JsonResponse({"data": "设备添加成功"})
        else:
            return JsonResponse({"data": "已存在该设备"})

    return out_func(request, in_func)


# 获取设备信息
def get_device(request):
    def in_func():
        page, = get_request(request, 'page')
        try:
            page = int(page)
        except:
            page = 1

        device = Device.objects.all()
        paginator = Paginator(device, page_num_each)
        result = paginator.page(page)
        result1 = []
        for item in result:
            temp = {}
            temp["id"] = item.id
            temp["mac"] = item.mac
            temp["name"] = item.name
            temp["floor"] = item.floor.floor
            temp["address"] = item.floor.building.address
            temp["position"] = item.position
            temp["type"] = item.type
            temp["building_type"] = item.floor.building.type.name

            result1.append(temp)

        return JsonResponse({"data": result1, "number": paginator.num_pages, "total": len(device)})

    return out_func(request, in_func)


# 设备下拉框
def select_device(request):
    def in_func():
        floor_id, = get_request(request, 'floor_id')
        device = list(Device.objects.filter(floor=floor_id).values('name', 'mac'))
        return JsonResponse({"data": device})

    return out_func(request, in_func)



