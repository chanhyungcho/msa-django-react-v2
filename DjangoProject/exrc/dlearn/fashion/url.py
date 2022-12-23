from django.urls import re_path as url

from exrc.dlearn.fashion import view as fashion_view


urlpatterns = [

    url(r'fashion', fashion_view.fashion)]