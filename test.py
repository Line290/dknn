
class b:
    def __init__(self, m):
        self.m = m
        self.sub_class = self.a(x=4)
    class a:
        def __init__(self, x):
            self.x = x
        def add(self, y):
            y += [self.x]

    def add(self):
        self.sub_class.add(self.m)



aa = b(m=[1,2])
print(aa.m) # [1, 2]
print(aa.sub_class.x) # 4
aa.add()
print(aa.m) # [5, 2]



