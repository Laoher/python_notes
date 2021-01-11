class point:
    def __init__(self,value,relationship):
        self.value = value
        self.relationship =relationship

class graph:
    known = []
    unknown = []
    def add(self,points):
        for point in points:
            self.unknown.append(point)
    def add_to_known(self,):
pointA = point("A",{"B":12,"G":14,"F":16})
pointB = point("B",{"A":12,"C":10,"F":7})
pointC = point("C",{"B":10,"D":3,"E":5,"F":6})
pointD = point("D",{"C":3,"E":4})
pointE = point("E",{"C":5,"D":4,"F":2,"G":8})
pointF = point("F",{"A":16,"B":7,"C":6,"E":2,"G":9})
pointG = point("G",{"A":14,"F":9,"E":8})


g = graph()
g.add([pointA,pointB,pointC,pointD,pointE,pointF,pointG])
print(g.unknown[0].value,g.unknown[0].relationship)