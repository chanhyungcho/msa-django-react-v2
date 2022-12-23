from django.urls import re_path as url
from multiplex.theatertickets import views

urlpatterns = [
    url(r'theatertickets', views.theatertickets)
]