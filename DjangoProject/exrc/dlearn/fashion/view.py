import json

from django.http import JsonResponse
from rest_framework.decorators import api_view

from exrc.dlearn.fashion.service import FashionService


@api_view(['GET', 'POST'])
def fashion(request):
    # if request.method == 'GET':
    #     print(f"######## test_num is {request.GET['test_num']} ########")
    #     return JsonResponse({'result': FashionService().service_model(int(request.GET['test_num']))})

    if request.method == 'POST':
        id = json.loads(request.body)  # json to dict
        print(f"######## Post id is {id} type is {type(id)} ########")
        a = FashionService().service_model(int(id))
        print(f'리턴결과 : {a}')
        return JsonResponse({'result': a})

    elif request.method == 'GET':
        print(f"######## Get id is {request.GET['id']} ########")
        return JsonResponse(
            {'result': FashionService().service_model(int(request.GET['id']))})

    else:
        print(f"######## test_num is None ########")



    """
    # 초기 버전
    if request.method == 'POST':
        data = request.data
        test_num = tf.constant(int(data['test_num']))
        result = FashionService().service_model([test_num])
        return JsonResponse({'result': result})
    # 신 버전
    if request.method == 'POST':
        data = json.loads(request.body)  # json to dict
        print(f"######## test_num is {data['test_num']} ########")
        return JsonResponse({'result': FashionService().service_model(int(data['test_num']))})
    """