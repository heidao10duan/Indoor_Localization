import collections
import json
import os
import time
import base64
import urllib.parse

from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from django.core.files.base import ContentFile
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

from RoomManagement.models import BuildingType, BuildingFloor, Building

# Create your views here.

page_num_each = 7


# 请求json中字段分解
def get_request(request, *args):
    if len(request.POST) <= 0:
        if type(request) == dict:
            req = request
        else:
            req = json.loads(request.body.decode())
            if type(req) == str:
                req = json.loads(req)
    else:
        req = request.POST
    result = []
    for item in args:
        if item not in req:
            result.append("")
        else:
            result.append(req[item])
    return result


def out_func(request, func):
    if request.method != 'POST':
        response = JsonResponse({"data": "请使用POST方法调用该接口"})
        response.status_code = 405
        return response
    try:
        return func()
    except Exception as e:
        print ("----ERROR-----------")
        print (str(e))
        response = JsonResponse({"data": str(e)})
        response.status_code = 500
        return response


def decode_base64(data):
    missing_padding = 4 - len(data) % 4
    if missing_padding:
        data += '='*missing_padding
    return base64.b64decode(data)


# 添加建筑物
def add_building(request):
    def in_func():
        name, avatar, address, center_coordinate, outline_coordinate, type = get_request(request, 'name', 'avatar',
                                                                                         'address',
                                                                                         'center_coordinate',
                                                                                         'outline_coordinate', 'type')

        print('-----name, address, center_coordinate, outline_coordinate, type----')
        print(name, address, center_coordinate, outline_coordinate, type)
        building_type_count = BuildingType.objects.filter(name=type).count()

        if building_type_count <= 0:
            return JsonResponse({"data": "不存在该建筑物类型"})

        building_count = Building.objects.filter(address=address).count()
        if building_count <= 0:
            data = decode_base64(avatar.split(',')[1])
            pic_suffix = avatar.split(',')[0].split('/')[1].split(';')[0]
            file_content = ContentFile(data)
            building = Building(name=name, address=address, center_coordinate=center_coordinate,
                                outline_coordinate=outline_coordinate, type=BuildingType(name=type))
            building.avatar.save("{0}_{1}.{2}".format(name, time.strftime("%Y%m%d%H", time.localtime()),pic_suffix), file_content)
            building.save()
            return JsonResponse({"data": "OK"})
        else:
            return JsonResponse({"data": "已存在该建筑物"})

    return out_func(request, in_func)


# 获取建筑物信息
def get_building(request):
    def in_func():
        page, = get_request(request, 'page')
        try:
            page = int(page)
        except:
            page = 1

        building = Building.objects.all()
        paginator = Paginator(building, page_num_each)
        pag_obj = paginator.page(page)
        result = []
        for i in range(len(pag_obj)):
            temp = {}
            temp['id'] = pag_obj[i].id
            temp['name'] = pag_obj[i].name
            temp['address'] = pag_obj[i].address
            temp['avatar'] = urllib.parse.unquote(pag_obj[i].avatar.url)
            result.append(temp)

        return JsonResponse({"data":list(result),"number":pag_obj.paginator.num_pages, "total":len(building)})

    return out_func(request, in_func)


# 模糊查询建筑物名称
def query_building(request):
    def in_func():
        address, = get_request(request,'address')
        building = list(Building.objects.filter(address__icontains=address))
        result = []

        for item in building:
            temp = {}
            temp["name"] = item.name
            temp["address"] = item.address
            temp["avatar"] = urllib.parse.unquote(item.avatar.url)
            result.append(temp)

        return JsonResponse({"data": result})

    return out_func(request, in_func)


# 建筑物下拉框
def select_building(request):
    def in_func():
        building = list(Building.objects.all())
        result = []

        for item in building:
            temp = {}
            temp['id'] = item.id
            temp['name'] = item.name
            temp['address'] = item.address
            temp['avatar'] = urllib.parse.unquote(item.avatar.url)
            result.append(temp)

        return JsonResponse({"data":result})

    return out_func(request, in_func)


# 删除建筑物
def delete_building(request):
    def in_func():
        building_id, = get_request(request, 'id')
        if isinstance(building_id,int):
            return JsonResponse({"data":"请上传正确的id"})

        building = Building.objects.filter(id=building_id).delete()
        if building[0] <= 0:
            return JsonResponse({"data":"不存在该建筑物"})

        return JsonResponse({"data": "建筑物删除成功"})

    return out_func(request, in_func)


# 添加建筑物类型：商场，车库等
def add_building_type(request):
    def in_func():
        building_type_name, = get_request(request, 'name')
        print("-----building_type_name----")
        print(building_type_name)
        # 插入到数据库中
        building_type_count = BuildingType.objects.filter(name=building_type_name).count()
        if building_type_count <= 0:
            building_type = BuildingType(name=building_type_name)
            building_type.save()
            return JsonResponse({"data": "OK"})
        # 若数据库存在，则提示已经有该建筑物类型
        else:
            return JsonResponse({"data": "已存在该建筑物类型"})

    return out_func(request, in_func)


# 获取建筑物类型
def get_building_type(request):
    def in_func():
        page, = get_request(request, 'page')
        try:
            page = int(page)
        except:
            page = 1

        building_type = BuildingType.objects.values()
        paginator = Paginator(building_type, page_num_each)
        result = paginator.page(page)
        return JsonResponse({"data": list(result), "number": result.paginator.num_pages,"total":len(building_type)})

    return out_func(request, in_func)


# 建筑物类型下拉框
def select_building_type(request):
    def in_func():
        building_type = list(BuildingType.objects.values())

        return JsonResponse({"data": building_type})

    return out_func(request, in_func)


# 删除建筑物类型
def delete_building_type(request):
    def in_func():
        name, = get_request(request, 'name')
        building_type = BuildingType.objects.filter(name=name).delete()

        if building_type[0] <= 0:
            return JsonResponse({"data": "不存在该建筑物类型"})

        return JsonResponse({"data": "建筑物类型删除成功"})

    return out_func(request, in_func)


# 添加楼层
def add_floor(request):
    def in_func():
        floor, building_id, uploadfile = get_request(request, 'floor', 'building', 'uploadfile')
        print("----floor, building_id-----")
        print(floor, building_id)
        building_count = Building.objects.filter(id=building_id).count()

        if building_count <= 0:
            return JsonResponse({"data": "不存在该建筑物id"})

        # 插入数据库，building_id是整数
        building_floor_count = BuildingFloor.objects.filter(floor=floor, building=Building(id=building_id)).count()

        if building_floor_count <= 0:
            data = decode_base64(uploadfile.split(',')[1])
            pic_suffix = uploadfile.split(',')[0].split('/')[1].split(';')[0]
            file_content = ContentFile(data)

            building_floor = BuildingFloor(floor=floor, building=Building(id=building_id))
            building_floor.floor_plan.save("{0}_{1}.{2}".format(floor, time.strftime("%Y%m%d%H", time.localtime()),pic_suffix), file_content)
            building_floor.save()
            return JsonResponse({"data": "OK"})
        else:
            return JsonResponse({"data": "该建筑物已添加该楼层"})

    return out_func(request, in_func)


# 获取楼层信息
def get_floor(request):
    def in_func():
        print ("888888888")
        print (request)
        print (request.POST)
        print (json.loads(request.body.decode()))
        page, = get_request(request, 'page')
        building_floor = list(BuildingFloor.objects.all())
        # building_floor = list(BuildingFloor.objects.values())

        try:
            page = int(page)
        except:
            page = 1

        paginator = Paginator(building_floor, page_num_each)
        result = paginator.page(page)
        result1 = []
        print (result)
        for item in result:
            print (item)
            temp = {}
            temp['id'] = item.id
            temp['floor'] = item.floor
            temp['name'] = item.building.name
            temp['address'] = item.building.address
            try:
                temp['avatar'] = str(urllib.parse.unquote(item.building.avatar.url))
            except:
                temp['avatar'] = "no avatar image"

            try:
                temp['floor_plan'] = str(urllib.parse.unquote(item.floor_plan.url))
            except:
                temp['floor_plan'] = "no image"
            result1.append(temp)
        return JsonResponse({"data":result1, "number":result.paginator.num_pages,"total":len(building_floor)})

    return out_func(request, in_func)


# 楼层下拉框
def select_floor(request):
    def in_func():
        building_id,  = get_request(request,'building_id')
        print ("----(****building_id---")
        print (building_id)
        building_floor = list(BuildingFloor.objects.filter(building_id=building_id).values('id','floor'))

        return JsonResponse({"data":building_floor})

    return out_func(request, in_func)


# 删除楼层信息
def delete_floor(request):
    def in_func():
        floor_id, = get_request(request, 'id')
        building_floor = BuildingFloor.objects.filter(id=floor_id).delete()

        if building_floor[0] <= 0:
            return JsonResponse({"data": "不存在该建筑物楼层"})

        return JsonResponse({"data": "该建筑物楼层删除成功"})

    return out_func(request, in_func)


# 根据id获取某建筑物某楼层
def get_map(request):
    def in_func():
        floor_id, = get_request(request, 'id')
        # floor_id = request.POST['id']
        print ('----floor_id----')
        print (floor_id)

        # 从文件夹中获取对应id的json数据
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print (BASE_DIR)
        map_filename = os.path.join(BASE_DIR + '/Building_Map/','{0}.json'.format(floor_id))

        if not os.path.exists(map_filename):
            return JsonResponse({"data":"该2D地图不存在"})
        with open(map_filename, "r",encoding='UTF-8') as f:
            result = json.loads(f.read())
        return JsonResponse(result)

    return out_func(request, in_func)


# 判断手机是否在建筑物内
def is_in_building(request):
    def in_func():
        lon, lat, floor_id = get_request(request, 'lon', 'lat', 'id')
        # 从数据库中取出floor_id对应的四个边的经纬度
        building_floor_lst = list(BuildingFloor.objects.filter(id=floor_id))

        if len(building_floor_lst) <= 0:
            return JsonResponse({"data": "没有该楼层"})

        temp_lst = building_floor_lst[0].building.outline_coordinate.split(',')
        lon_lst = [float(item.split(" ")[0]) for item in temp_lst if len(item.split(" "))==2]
        lat_lst = [float(item.split(" ")[1]) for item in temp_lst]
        max_lon = max(lon_lst)
        print ("max_lon:", max_lon)
        min_lon = min(lon_lst)
        print ("min_lon:", min_lon)
        max_lat = max(lat_lst)
        print ("max_lat:", max_lat)
        min_lat = min(lat_lst)
        print ("min_lat:", min_lat)

        # 判断lon,lat是否在上面的四个经纬度内
        if lon >= min_lon and lon <= max_lon and lat >= min_lat and lat <= max_lat:
            return JsonResponse({"data": "OK"})
        else:
            return JsonResponse({'data': "不在该建筑物内"})

    return out_func(request, in_func)