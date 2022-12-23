from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from exrc.nlp.samsung_report.models import Controller, Service


@api_view(['GET'])
def nlp(request):
    return JsonResponse({'result':
                             Controller().data_analysis()})
