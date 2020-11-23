import inspect


class A(object):
    def __init__(self, name, x=2, y=3):

        for k, v in dict(locals()).items():
            self.__setattr__(k, v)


a = A('a', x=5)
print(a.x)
