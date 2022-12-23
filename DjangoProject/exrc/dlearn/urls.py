from django.urls import re_path as url
from exrc.dlearn.iris import view as iris_view
from exrc.dlearn.fashion import view as fashion_view
from exrc.dlearn.mnist import view as mnist_view

urlpatterns = [
    url(r'iris', iris_view.iris),
    url(r'fashion', fashion_view.fashion),
    url(r'mnist', mnist_view.mnist)
]