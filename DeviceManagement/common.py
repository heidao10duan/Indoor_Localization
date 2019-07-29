import json

from django.http import JsonResponse


# -----------------------------------------

page_num_each = 3


# 请求json中字段分解
def get_request(request, *args):
    if type(request) == dict:
        req = request
    else:
        req = json.loads(request.body.decode())
        if type(req) == str:
            req = json.loads(req)
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