import math
import sys


def nav_info_13():
    shops1 = {
        # "鹿岛普通生活": (241,163),
        # "拉.贝缇": (310,164),
        # "满记甜品":(200,200),
        # "TATA女鞋店": (197,290),
        # "BASTO": (198,347),
        # "UlifeStyle": (196,405),
        # "7m": (196,482),
        # "MLB": (257,316),
        # "木九十":(420,164),
        # "eifini": (454,154),
        # "ONE MORE": (489,144),

        "广岛森林": [1043,141],
        "鲜之味": [1007,236],
        "闺蜜坊":[1007,236],
        "瑞幸咖啡": [962,320]

    }

    specific_shop1 = {"广岛森林":[[1043,141],[1028,135]],
                      "鲜之味":[[1007,236],[953,217]],
                      "闺蜜坊": [[1007,236],[973,226]],
                      "瑞幸咖啡": [[962,320],[964,299]]
                      }

    road1 = [
        [
          1043,
          141
        ],
        [
          1007,
          236
        ],
        [
          962,
          320
        ],
        [
          938,
          355
        ],
        [
          925,
          377
        ],
        [
          790,
          534
        ],
        [
          675,
          631
        ],
        [
          491,
          711
        ],
        [
          453,
          728
        ],
        [
          425,
          742
        ],
        [
          397,
          750
        ],
        [
          366,
          761
        ],
        [
          189,
          795
        ],
        [
          189,
          793
        ]
      ]

    road_num = 1
    result = []
    result.append({
        "shop": shops1,
        "specific_shop": specific_shop1,
        "road": road1,
        "road_point": [],
        "point_roads": {}
    })

    return road_num, result


# floor_id对应的道路信息
def nav_info_3():
    # 特别说明：岔路口，俩条路的话，可以直接用岔路口，多条路还的重新考虑一下
    print ("***********************")
    print (sys._getframe().f_code.co_name)
    # 各个商店的名称和路线上对应的坐标点
    shops1 = {"JEEP": (25, 21), "new balance": (35, 21), "ONLY女装店": (35, 21), "Zara服装店": (42, 21),
              "CALVIN KLEIN PERFORMANCE": (42, 21), "H:CONNECT": (75, 37), "联想4S店": (47, 55),
              "雅戈尔": (43, 57), "Mirosoft": (41, 58), "BRING DREAMS": (36, 59), "博士眼镜": (36, 59), "HUAWEI": (30, 60),
              "时代印象": (25, 59), "MJstyle": (25, 59),
              "吉盟珠宝": (25, 52), "inxx": (25, 48), "TISSOT": (25, 42), "UNIQLO": (25, 42), "innisfree": (25, 36),
              "屈臣氏": (25, 30)}
    # 商店路线上坐标点到商店门口的路线
    specific_shop1 = {"JEEP": [(25, 21),(25,18)], "H:CONNECT": [(76, 38), (77, 39)], "ONLY女装店": [(35, 21), (35, 30)],
                      "联想4S店": [(47, 55)],"Zara服装店": [(42, 21),(45,26)]}
    shops2 = {"BreadTalk": (89, 6), "元祖": (105, 11), "beauty factory boutique": (105, 15), "好又来": (103, 20),
              "芭比馒头": (103, 22), "香络天": (102, 24), "一品香川菜馆": (100, 28),
              "阿蚝海鲜焖面": (95, 34), "霸王茶姬": (93, 37), "米小姐的餐厅": (92, 39), "映佳": (86, 48), "西洋屋": (84, 50),
              "跃华茶业": (80, 54),
              "奢时": (79, 56), "Oyeh": (76, 59), "空铺": (3, 27), "WHO.A.U": (66, 27)}
    road1 = [(25, 21), (31, 21), (35, 21), (42, 21), (53, 19), (65, 18), (78, 20), (84, 21), (87, 21), (81, 27),
             (79, 31), (77, 35), (75, 37), (72, 40), (67, 43), (59, 50),
             (52, 53), (47, 55), (43, 57), (41, 58), (36, 59), (36, 59), (30, 60), (25, 59), (25, 52), (25, 48),
             (25, 42), (25, 36), (25, 30)]
    # 路线上的岔路口坐标点
    road1_points = [(84, 21), (25, 21), (87, 21)]
    # 岔路口的路线
    point_roads1 = {(84, 21): [(84, 21), (89, 15), (89, 6)], (25, 21): [(25, 21), (25, 6)],
                    (87, 21): [(87, 21), (101, 26)]}

    road2 = [(66, 27), (59, 27), (52, 27), (3, 27), (3, 6), (25, 6), (41, 6), (53, 6), (66, 6), (84, 6), (89, 6),
             (105, 6), (105, 11), (105, 15), (103, 20), (103, 22), (102, 24), (101, 26), (100, 28), (98, 31), (95, 34),
             (93, 37), (92, 39), (87, 46), (86, 48), (84, 50),
             (80, 54), (79, 56), (76, 59)]
    road2_points = [(89, 6), (25, 6), (101, 26)]
    point_roads2 = {(89, 6): [(89, 6), (89, 15), (84, 21)], (25, 6): [(25, 6), (25, 21)],
                    (101, 26): [(101, 26), (87, 21)]}
    road_num = 2
    result = []
    result.append({"shop":shops1,
                   "specific_shop":specific_shop1,
                   "road":road1,
                   "road_point":road1_points,
                   "point_roads":point_roads1})
    result.append({
        "shop": shops2,
        "specific_shop": {},
        "road": road2,
        "road_point": road2_points,
        "point_roads": point_roads2
    })

    return road_num, result


def main(floor_id, startName, endName):
    print (floor_id)
    road_num, result = eval("nav_info_{0}".format(floor_id))()
    start_index = -1
    end_index = -1

    # 查找起点和终点所在的路径
    for i in range(len(result)):
        item = result[i]
        road = item["road"]
        shops = item["shop"]
        print ("***********shops")
        print (shops)
        specific_shop = item["specific_shop"]

        # 起点和终点在一条路上
        if startName in shops and endName in shops:
            startLoc = shops[startName]
            endLoc = shops[endName]
            path_lst = []

            # 起点到门口的路线
            if startName in specific_shop:
                path_lst.extend(specific_shop[startName])

            # 起点到终点的路线
            path_lst.extend(nav(road, startLoc, endLoc))

            # 终点到门口的路线
            if endName in specific_shop:
                path_lst.extend(specific_shop[endName])

            print ("***path")
            print (path_lst)
            return path_lst

        if startName in shops:
            start_index = i

        if endName in shops:
            end_index = i

    # 起点或终点不存在
    if start_index == -1 or end_index == -1:
        print ("**************")
        print ("不存在该起点或终点")
        return []

    # 起点和终点在不同的路线上
    # **start
    path_lst = []
    start_nav = result[start_index]
    startLoc = start_nav["shop"][startName]
    point_dis = []

    # 起点到门口
    path_lst.extend(start_nav['specific_shop'][startName])

    # 起点到最近的岔路口
    for point in start_nav["road_point"]:
        point_dis.append(math.sqrt((point[0] - startLoc[0]) ** 2 + (point[1] - startLoc[1]) ** 2))

    min_point = min(point_dis)

    # 起点到岔路口的路线
    min_index = point_dis.index(min_point)
    endLoc = start_nav["road_point"][min_index]
    path_lst.extend(nav(start_nav["road"], startLoc, endLoc))

    # 岔路口的路线
    path_lst.extend(start_nav["point_roads"][endLoc])

    # 岔路口到终点的路线
    end_nav = result[end_index]
    endLoc = end_nav["shop"][endName]
    path_lst.extend(nav(end_nav["road"], start_nav["point_roads"][start_nav["road_point"][point_dis.index(min_point)]][-1], endLoc))

    # 终点到门口
    path_lst.extend(end_nav['specific_shop'][endName])
    return path_lst


def nav(roads, startLoc, endLoc):
    path_lst = []
    flag = True
    # 起点和终点是同一个点
    if startLoc == endLoc:
        return [startLoc]

    for item in roads:
        if flag:
            # 路径上找到了起点或终点，开始寻路
            if item == startLoc or item == endLoc:
                path_lst.append(item)
                flag = False
                continue
        else:
            # 路径上找到终点或起点，结束寻路
            if item == startLoc or item == endLoc:
                path_lst.append(item)
                break

            path_lst.append(item)

    # 正向路程太长，就反向，因为是list
    if len(path_lst) > len(roads)/2:
        start_index = roads.index(startLoc)
        end_index = roads.index(endLoc)
        temp1 = roads[:start_index+1]
        temp1.reverse()
        temp2 = roads[end_index:]
        temp2.reverse()
        path_lst = temp1 + temp2

    if startLoc != path_lst[0]:
        path_lst.reverse()

    return path_lst


