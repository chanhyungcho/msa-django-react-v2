from django.urls import re_path as url
from multiplex.movies import views

urlpatterns = [
    url(r'movies', views.movies)
]