import json
from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from exrc.webcrawler.navermovie.services import ScrapService

@api_view(['GET'])
def crawler(request):
    return JsonResponse({'result':
                ScrapService().naver_movie_review()})
    #print(f'######################{type(resp)}########################{resp}')


