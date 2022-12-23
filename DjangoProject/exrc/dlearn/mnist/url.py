from django.urls import re_path as url
from exrc.dlearn.mnist import view as mnist_view

urlpatterns = [
    url(r'mnist', mnist_view.mnist)
]