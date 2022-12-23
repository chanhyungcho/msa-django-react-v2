from django.shortcuts import render

from django.http import JsonResponse
from rest_framework.decorators import api_view

@api_view(['POST'])
def signup(request):
    user_info = request.data
    email = user_info['email']
    nickname = user_info['nickname']
    password = user_info['password']
    print(f'리액트에서 보낸 데이터: {user_info}')
    print(f'넘어온 이메일: {email}')
    print(f'넘어온 닉네임: {nickname}')
    print(f'넘어온 비밀번호: {password}')
    return JsonResponse({'회원가입 결과':'성공!'})