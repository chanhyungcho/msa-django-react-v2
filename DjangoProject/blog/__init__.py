'''from mpg import MpgService,MPG_MENUS
def my_menu(ls):
    for i, j in enumerate(ls):
        print(f"{i}. {j}")
    return input('메뉴선택: ')
if __name__ == '__main__':
    t = MpgService()
    while True:
        menu = my_menu(MPG_MENUS)
        if menu == '0':
            print("종료")
            break
        else:
            try:
                MPG_MENUS[menu](t)
            except KeyError:
                print(" ### Error ### ")
'''