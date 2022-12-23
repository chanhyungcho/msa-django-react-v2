from django.urls import re_path as url

from exrc.stroke import view as view_stroke

urlpatterns = [
    url(r'stroke', view_stroke.stroke)
]