from django.urls import re_path as url
from multiplex.theaters import views

urlpatterns = [
    url(r'theaters', views.theaters)
]