from django.urls import re_path as url
from multiplex.cinemas import views

urlpatterns = [
    url(r'cinemas', views.cinemas)
]