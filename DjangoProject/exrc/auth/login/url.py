from django.urls import re_path as url

from exrc.auth.login import view as view_login


urlpatterns = [
    url(r'login', view_login.login)
]