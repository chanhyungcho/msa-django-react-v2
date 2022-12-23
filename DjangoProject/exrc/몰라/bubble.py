import random

class Bubble(object):
    def __init__(self):
        pass

    def extract_choice(self):
        rl = random.sample(range(10,100), 10)
        print(rl)
        for i in range(len(rl)-1):
            for j in range(len(rl)-1):
                if rl[j] > rl[j+1]:
                    rl[j], rl[j+1] = rl[j+1], rl[j]
        print("*"*30)
        print(rl)

    @staticmethod
    def main():
        bubble = Bubble()
        bubble.extract_choice()

Bubble.main()