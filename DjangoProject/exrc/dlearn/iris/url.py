from django.urls import re_path as url
from exrc.dlearn.iris import view as iris_view


urlpatterns = [
    url(r'iris', iris_view.iris)
    ]