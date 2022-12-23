from django.urls import re_path as url

from exrc.auth.signup import view as view_signup


urlpatterns = [
    url(r'signup', view_signup.signup)
]