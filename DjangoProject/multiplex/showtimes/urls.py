from django.urls import re_path as url
from multiplex.showtimes import views

urlpatterns = [
    url(r'showtimes', views.showtimes)
]