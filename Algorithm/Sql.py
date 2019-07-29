import pymysql.cursors


class ConnectDb(object):
    def __init__(self):
        # self.conn = pymysql.Connect(
        # host='172.16.201.230',
        # port=3306,
        # user='root',
        # passwd='root',
        # db='indoor_localization_own',
        # charset='utf8'
        # )
        self.conn = pymysql.Connect(
            host='127.0.0.1',
            port=3306,
            user='root',
            passwd='123456',
            db='test1',
            charset='utf8'
        )

    def select_table(self,tablename, wherename, wherevalue, *args):
        cursor = self.conn.cursor()
        sql = "SELECT {0} FROM `{1}` WHERE {2}='{3}';".format(",".join(args),tablename,wherename, wherevalue)
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result

    def insert_table(self, data):
        result = ""
        for i in range(len(data)):
            tags = self.select_table("device_management_tag", "mac", data[i][0], "id","room_id")
            if tags == ():
                print ("不存在该tag")
                continue
            # employees = self.select_table("employee_management_employee", "tag_id", tags[0][0],"id")
            # if employees == ():
            #     print ("不存在该tag对应的员工")
            #     continue
            # 这边还有点问题
            tmp = list(data[i])
            tag_id, room_id = tags[0]
            print (room_id)
            # employee_id = employees[0][0]
            employee_id =1
            tmp.extend((tag_id, employee_id, room_id))
            result += str(tuple(tmp)) + ','
        # 根据mac从表employee_management_employee中取出Employee_id
        # 根据mac从表device_management_tag中取出room_id和tag_id
        sql = "INSERT INTO `location_info_taghistory`  (`name`, `position`, `timestamp`, `tag_id`, `Employee_id`, `room_id`) VALUES {0}".format(result[:-1])
        print ("*sql*sql:", sql)
        cursor = self.conn.cursor()
        cursor.execute(sql)
        sql = "REPLACE INTO `location_info_taglocation` (`name`, `position`, `timestamp`, `tag_id`, `Employee_id`, `room_id`) VALUES {0}".format(result[:-1])
        cursor.execute(sql)
        self.conn.commit()
        print ("insert success")
        cursor.close()
        self.conn.close()


# a = ConnectDb()
# result = a.select_table("device_management_tag","mac","aa", "id","room_id")
# print ("**")
# tag_id,room_id = result[0]
# print ("id:{0}, room_id:{1}".format(tag_id, room_id))
# result1 = a.select_table("employee_management_employee", "tag_id", tag_id, "id")
# print (type(result), result, result==())
# print (result1)
