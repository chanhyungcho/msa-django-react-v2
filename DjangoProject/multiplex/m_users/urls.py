from django.urls import re_path as url
from multiplex.m_users import views

urlpatterns = [
    url(r'cinemas', views.m_users)
]