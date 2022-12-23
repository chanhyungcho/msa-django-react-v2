from django.http import JsonResponse
from rest_framework.decorators import api_view

from exrc.stroke.model import StrokeModel


@api_view(['POST'])
def stroke(request):
    StrokeModel().hook()
    print(f'Enter Stroke with {request}')
    return JsonResponse({'Response Test':'SUCCESS'})
