from cvxpy import *

# Example in DeepSRGR, finding the boundaries of neurons

x1 = Variable()
x2 = Variable()
x3 = Variable()
x4 = Variable()
y1 = Variable()
y2 = Variable()
t = Variable()
# Create two constraints.
constraints = [x1 >= -1, x1 <= 1, x2 >= -1, x2<=1,x3==x1-x2,  x4== x1+x2+2.5, y1>=0,y1<=0.5*x3+1,y1>=x3,y2>=x4,y2<=x4,y2-y1<=0, t==y2-y1]

obj1 = Minimize(x1)
obj2 = Maximize(x1)

obj3 = Minimize(x2)
obj4 = Maximize(x2)

obj5 = Minimize(x3)
obj6 = Maximize(x3)

obj7 = Minimize(x4)
obj8 = Maximize(x4)

# obj9 = Minimize(y1)
# obj10 = Maximize(y1)

obj11 = Minimize(t)
obj12 = Maximize(t)

prob = Problem(obj1, constraints)
prob2 = Problem(obj2, constraints)

prob3 = Problem(obj3, constraints)
prob4 = Problem(obj4, constraints)


prob5 = Problem(obj5, constraints)
prob6 = Problem(obj6, constraints)

prob7 = Problem(obj7, constraints)
prob8 = Problem(obj8, constraints)

# prob9 = Problem(obj9, constraints)
# prob10 = Problem(obj10, constraints)


prob11 = Problem(obj11, constraints)
prob12 = Problem(obj12, constraints)


prob.solve() # Returns the optimal value.
prob2.solve()
prob3.solve()
prob4.solve()
prob5.solve()
prob6.solve()
prob7.solve()
prob8.solve()
# prob9.solve()
# prob10.solve()
prob11.solve()
prob12.solve()

number = 1.4700706177116257e-10
print('%.0000000f' % number)

print("status:", prob.status)
print("optimal value ", prob.value)
print("optimal value2 ", prob2.value)
print("optimal value3 ", prob3.value)
print("optimal value4 ", prob4.value)
print("optimal value5 ", prob5.value)
print("optimal value6 ", prob6.value)
print("optimal value7 ", prob7.value)
print("optimal value8 ", prob8.value)
# print("optimal value9 ", prob9.value)
# print("optimal value10 ", prob10.value)

print("optimal value11 ", prob11.value)
print("optimal value12 ", prob12.value)


print("optimal var", x1.value, y1.value)