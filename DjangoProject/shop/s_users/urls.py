from django.urls import re_path as url
from shop.s_users import views

urlpatterns = [
    url(r'products', views.s_users)
]