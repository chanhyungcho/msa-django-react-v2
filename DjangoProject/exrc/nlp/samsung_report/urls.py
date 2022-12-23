from django.urls import re_path as url

from exrc.nlp.samsung_report import views as view_samsung_report

urlpatterns = [
    url(r'samsung-report', view_samsung_report.nlp)
]