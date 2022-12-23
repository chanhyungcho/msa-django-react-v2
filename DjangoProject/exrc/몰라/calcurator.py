class Calcurator(object):
    def __init__(self,num,op,num2):
        self.num = num
        self.op = op
        self.num2 = num2

    @staticmethod
    def new_calc():
        num = int(input("숫자1: "))
        op = (input("+,-,%,/,*"))
        num2 = int(input("숫자2: "))
        return Calcurator(num, op, num2)

    def execute_info(self):
        num = self.num
        op = self.op
        num2 = self.num2
        if op == "+":
            result =  num + num2
        elif op == "-":
            result = num - num2
        elif op == "/":
            result = num / num2
        elif op == "*":
            result = num * num2
        elif op == "%":
            result = num % num2
        return result

    def print_info(self):
        print(f"{self.num} {self.op} {self.num2} = {self.execute_info()} ")


    @staticmethod
    def get_calc(ls):
        [i.print_info() for i in ls]


    @staticmethod           #삭제가 안됨. 인덱스 기준으로 보는 게 아니라 다른걸로 해야하나
    def delect_calc(ls,num):
        for i in ls:
            if i.num == num:
                del ls[i]

    @staticmethod
    def print_menu():
        print("1. 계산 등록")
        print("2. 계산 출력")
        print("3. 계산 삭제")
        print("4. 종료")
        return int(input("메뉴 선택: "))


def main():
    ls = []
    while True:
        menu = Calcurator.print_menu()
        if menu == 1:
            print("1. 계산등록")
            ls.append(Calcurator.new_calc())
        elif menu == 2:
            print("2. 계산출력")
            Calcurator.get_calc(ls)
        elif menu == 3:
            print("3. 계산삭제")
            Calcurator.delect_calc(ls, input("삭제할 결과의 첫번째숫자: "))
        elif menu == 4:
            print("4. 종료")
            break
        else:
            print("잘못된 접근입니다")


main()