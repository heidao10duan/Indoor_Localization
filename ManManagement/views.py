import csv
import json
import os
import threading

from django.shortcuts import render
from django.http import JsonResponse

from Algorithm.navigation_algorithm import start_process
from Algorithm.localization import IndoorLocalization
from .models import MapGrid
from RoomManagement.models import BuildingFloor
from DeviceManagement.models import DeviceGroup, Device
from RoomManagement.models import BuildingFloor

# Create your views here.


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
    print ("_____***req")
    print (req)
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
        response = JsonResponse({"data": str(e)})
        response.status_code = 500
        return response


# 根据rssi计算位置
# ============需要修改============
def get_location(request):
    def in_func():
        appid, secret,js_code, floor_id, rssi_data = get_request(request,'appid', 'secret','js_code', 'floor_id', 'rssi_data')
        print ("---request---")
        print (appid, secret,js_code, floor_id, rssi_data)
        # 信号强度按从强到若，取出前10个，好到同一分组的数据，用三点定位得到结果
        device_dict = {}

        for item in rssi_data:
            device_id = item['deviceId'].replace(':','')
            print (device_id)
            RSSI = item['RSSI']
            timestamp = item['timestamp']
            device = list(Device.objects.filter(mac=device_id))
            if len(device) <= 0:
                continue

            device_group = device[0].group

            if device_id not in device_dict:
                device_dict.setdefault(device_id,{})
                device_dict[device_id].setdefault("rssi",[])
                device_dict[device_id].setdefault("timestamp",[])
                device_dict[device_id].setdefault('group', device_group)

            device_dict[device_id]['rssi'].append(RSSI)
            device_dict[device_id]['timestamp'].append(timestamp)

        print ("-------device_dict--------")
        print (device_dict)

        # 按照均值排序，取前10个，若最终找不到一个组有至少3个deviceid的情况，则可用全部的
        # device_lst = sorted(device_dict, key=lambda x: x)
        device_lst = sorted(device_dict.items(), key=lambda x: sum(x[1]['rssi'])/len(x[1]['rssi']), reverse=True)
        device_lst = device_lst[:10]
        print ("-----device_lsit____")
        print (device_lst)

        group_dict = {}

        for item in device_lst:
            if item[1]['group'] not in group_dict:
                group_dict.setdefault(item[1]['group'], 0)
            group_dict[item[1]['group']] +=1
        print("----------group_dict-----------")
        print(group_dict)

        groups = sorted(group_dict.items(), key=lambda x:x[1], reverse=True)
        group_id = groups[0][0]
        print (group_id)
        group_dict = {}
        # 取出group_id组的数据，进行定位
        for item in device_lst:
            if item[1]['group'] == group_id:
                group_dict[item[0]] = item[1]

        print ("----------group_dict-----------")
        print (group_dict)

        indoor_location = IndoorLocalization()
        indoor_location.start_process(appid, group_dict)


        # 查找每个deviceid所在的组，看每个组包含的deviceid数量，超过三个则进行定位

        device_group = {}
        # 调用定位接口

        return JsonResponse({"data": "OK"})

    return out_func(request, in_func)


# 比较俩个值的大小
def compare_value(value1, value2):
    if value1 > value2:
        return value1, value2
    return value2, value1


# 根据楼层和起始点位置确定导航路径
def get_navigation(request):
    def in_func():
        floor_id, start_x, start_y, geo_id = get_request(request, 'floor_id', 'start_x', 'start_y', 'geo_id')
        start_x, start_y = int(start_x), int(start_y)
        # floor_id, start_x, start_y, geo_name = int(request.POST['floor_id']), int(request.POST['start_x']), int(
        #     request.POST['start_y']), request.POST['geo_name']
        # 判断起点和终点是否在障碍物内，若在
        # 1、从起点导航到障碍物的门口
        # 2、从起点的门口导航到终点的门口
        # 3、从终点的门口导航到终点
        # 提前设计好哪些栅格是被阻挡的，调用接口计算结果
        # 根据floor_id从数据库中获取被主档的栅格
        map_grid = list(MapGrid.objects.filter(floor=floor_id))
        if len(map_grid) <= 0:
            return JsonResponse({"data": "不存在该楼层的栅格图"})

        # 根据floor_id从数据库中获取栅格地图的宽和高
        building_floor = list(BuildingFloor.objects.filter(id=floor_id))

        if len(building_floor) <= 0:
            return JsonResponse({"data": "不存在该楼层"})

        border_x = building_floor[0].height
        border_y = building_floor[0].width

        map_grid1 = list(MapGrid.objects.filter(floor=floor_id, id=geo_id))
        end_x = map_grid1[0].door_x
        end_y = map_grid1[0].door_y

        flag_start_point = True
        start_navigation_temp = []
        obstacle_lst = []
        navigation_temp = []

        for grid in map_grid:
            if flag_start_point:
                x_max, x_min = compare_value(grid.start_x, grid.end_x)
                y_max, y_min = compare_value(grid.end_x, grid.end_y)

                if start_y >= x_min and start_y <= x_max and start_x >= y_min and start_x <= y_max:
                    # 从数据库中获取门口的坐标
                    print("***********障碍物内不**********")
                    start_navigation_temp = start_x, start_y, grid.door_x, grid.door_y, border_y, border_x, []
                    flag_start_point = False

            obstacle_lst.append([grid.start_x, grid.start_y, grid.end_x, grid.end_y])

        if len(start_navigation_temp) != 0:
            navigation_temp.extend([start_navigation_temp[2], start_navigation_temp[3]])
        else:
            navigation_temp.extend([start_x, start_y])

        navigation_temp.extend([end_x, end_y])
        navigation_temp.extend([border_x, border_y, obstacle_lst])

        print("********navi")
        print(navigation_temp)
        print(start_navigation_temp)

        result = []
        result1 = []

        if len(start_navigation_temp) != 0:
            print("**start_process")
            temp_result = start_process(start_navigation_temp)
            result1 = temp_result
        print("**process")
        temp_result = start_process(navigation_temp)

        result.extend(result1)
        result.extend(temp_result)
        return JsonResponse({"data": result})

    return out_func(request, in_func)


# 获取障碍物的名称
def get_destname(request):
    def in_func():
        floor_id, = get_request(request, 'floor_id')
        map_grid = list(MapGrid.objects.filter(floor=BuildingFloor(id=floor_id)).values('id','geo_name'))
        return JsonResponse({"data": map_grid})

    return out_func(request, in_func)


def get_train_data(request):
    def in_func():
        appid, secret, js_code, floor_id, rssi_data = get_request(request, 'appid', 'secret', 'js_code', 'floor_id',
                                                                  'rssi_data')
        print("---request---")
        global data_lst
        data_lst = rssi_data
        data_path = r"C:\Users\Juan\Desktop\tagdata"
        tag_name = "MiniBeacon_00679"
        distance = 6
        direction = "N"
        file_path = os.path.join(data_path, "{0}-{1}m-{2}.csv".format(tag_name, distance, direction))

        for item in rssi_data:
            if item['name'] == "MiniBeacon_00679":
                rssi = item['RSSI']
                timestamp = item['timestamp']
                if not os.path.exists(file_path):
                    out = open(file_path, 'a', newline='')
                    # 设定写入模式
                    csv_write = csv.writer(out, dialect='excel')
                    # 写入具体内容
                    csv_write.writerow(["rssi",	"timestamp", "tag", "distance","direction"])
                else:
                    out = open(file_path, 'a', newline='')
                    # 设定写入模式
                    csv_write = csv.writer(out, dialect='excel')
                csv_write.writerow([rssi, timestamp, tag_name, distance, direction])
                print("write over")
        return JsonResponse({"data": "OK"})

    return out_func(request, in_func)


# 路径导航
def get_navigationtest(request):
    floor_id, start_name, end_name = get_request(request, 'floor_id', 'start_name', 'end_name')

    from Algorithm.navigation_alg import main
    return JsonResponse({"data":main(floor_id, start_name, end_name)})


