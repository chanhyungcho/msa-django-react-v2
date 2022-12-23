import numpy as np
import pandas as pd
import random
import string
from sqlalchemy import create_engine

from exrc.algorithms.lambdas import lambda_string, lambda_k_name, lambda_number


class B_userService(object):
    def __init__(self):
        global engine #jdbc:mysql://localhost:3306/mydb
        engine = create_engine(
            "mysql+pymysql://root:root@localhost:3306/mydb",
            encoding='utf-8')

    def insert_users(self):
        dc =self.create_user()
        ls = self.create_users(dc)
        df = self.change_df_to_users(ls)
        df.to_sql(name='b_users',
                  if_exists='append',
                  con=engine,
                  index=False)

    def create_user(self) -> {}:
        # string_pool = string.ascii_lowercase
        # blog_userid = random.randint(9999,
        #                              99999)  # model의 dtype이 숫자인 AutoField로 돼있어서 임시로 수정되면 "blog_userid= ''.jo
        # # in(random.sample(string_pool, 5))"로 변경(email도 email = blog_userid + "@naver.com"로 변경)

        email = str(lambda_string(4)) + "@naver.com"
        nickname = lambda_k_name(2)
        password = 0
        print(email)
        print(nickname)
        print(password)
        return email,nickname,password




    def create_users(self)-> []:
        n = 5
        email = ''
        golbang = '@google.com'
        password = lambda_number(4)
        nickname = lambda_k_name(2)
        data = []
        # email_list = list()

        for i in range(100):
            str(lambda_string(4)) + "@naver.com"

            data.append([email,nickname,password])
            email=''

        df = pd.DataFrame(data,columns=['email','nickname','password'])
        print(df.duplicated(['email']))
        print(df)
        return df



    def change_df_to_users(self)-> {}:
        pass

    def get_users(self)-> {}:
        pass





        # email_list = b_users.objects.values_list("email", flat=True)
        # if b_users.objects.filter(email=data["email"]).exists():
        #     return ~~


    def drop_df(self):
        df = self.insert_users()
        spac = df[df['email'].str.contains('@')].index
        df.drop(spac,inplace=True)


stroke_menu = ["Exit",  # 0
               "create_users",# 1
               "show_csv", #2
               "create_user" #3
               ]
stroke_lambda = {
    "1": lambda x: x.create_users(),
    "2": lambda x: x.insert_users(),
    "3": lambda x: x.create_user(),


}
if __name__ == '__main__':
    stroke = B_userService()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(stroke_menu)]
        menu = input('메뉴선택: ')
        if menu == '0':
            print("종료")
            break
        else:
            try:
                stroke_lambda[menu](stroke)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message')
                else:
                    print("Didn't catch error message")




