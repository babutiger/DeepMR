from cvxpy import *

# Example in DeepMR, finding the boundaries of neurons

x1 = Variable()
x2 = Variable()
x3 = Variable()
x4 = Variable()
x5 = Variable()
x6 = Variable()
x7 = Variable()

l1 = Variable()
l2 = Variable()
l3 = Variable()

t1 = Variable()
t2 = Variable()

u1 = Variable(boolean=True)
u2 = Variable(boolean=True)

constraints = [x1 >= -1, x1 <= 1, x2 >= -1, x2<=1, x3<=1, x3 >= -1, x4 <= 1, x4 >= -1,  x5 == x1+x2+x3+x4+5.5, x6 == x1+x2-x3-x4, x7 == x1-x2-x3+x4,
l1 >= x5, l1 <= x5, l2 >= 0, l2 <= 0.5*x6+2, l3 >= 0, l3 <= 0.5*x7+2,
t1==l2-l1, t2==l3-l1,
t1 >= -10000*(1-u1), t2 >= -10000*(1-u2),
u1 >= 1-u2
]

obj1 = Minimize(x1)
obj2 = Maximize(x1)

obj3 = Minimize(x2)
obj4 = Maximize(x2)

obj5 = Minimize(x3)
obj6 = Maximize(x3)

obj7 = Minimize(x4)
obj8 = Maximize(x4)

obj9 = Minimize(x5)
obj10 = Maximize(x5)

obj11 = Minimize(x6)
obj12 = Maximize(x6)

obj13 = Minimize(x7)
obj14 = Maximize(x7)

obj15 = Minimize(l1)
obj16 = Maximize(l1)

obj17 = Minimize(l2)
obj18 = Maximize(l2)

obj19 = Minimize(l3)
obj20 = Maximize(l3)

obj21 = Minimize(t1)
obj22 = Maximize(t1)

obj23 = Minimize(t2)
obj24 = Maximize(t2)


prob1 = Problem(obj1, constraints)
prob2 = Problem(obj2, constraints)

prob3 = Problem(obj3, constraints)
prob4 = Problem(obj4, constraints)

prob5 = Problem(obj5, constraints)
prob6 = Problem(obj6, constraints)

prob7 = Problem(obj7, constraints)
prob8 = Problem(obj8, constraints)

prob9 = Problem(obj9, constraints)
prob10 = Problem(obj10, constraints)

prob11 = Problem(obj11, constraints)
prob12 = Problem(obj12, constraints)

prob13 = Problem(obj13, constraints)
prob14 = Problem(obj14, constraints)

prob15 = Problem(obj15, constraints)
prob16 = Problem(obj16, constraints)

prob17 = Problem(obj17, constraints)
prob18 = Problem(obj18, constraints)

prob19 = Problem(obj19, constraints)
prob20 = Problem(obj20, constraints)

prob21 = Problem(obj21, constraints)
prob22 = Problem(obj22, constraints)

prob23 = Problem(obj23, constraints)
prob24 = Problem(obj24, constraints)


prob1.solve() # Returns the optimal value.
prob2.solve()
prob3.solve()
prob4.solve()
prob5.solve()
prob6.solve()
prob7.solve()
prob8.solve()
prob9.solve()
prob10.solve()
prob11.solve()
prob12.solve()
prob13.solve()
prob14.solve()
prob15.solve()
prob16.solve()
prob17.solve()
prob18.solve()
prob19.solve()
prob20.solve()
prob21.solve()
prob22.solve()
prob23.solve()
prob24.solve()


number = 1.4700706177116257e-10
print('%.0000000f' % number)

print("status:", prob1.status)
print("optimal value1--x1min ", prob1.value)
print("optimal value2--x1max ", prob2.value)
print("optimal value3--x2min ", prob3.value)
print("optimal value4--x2max ", prob4.value)
print("optimal value5--x3min ", prob5.value)
print("optimal value6--x3max ", prob6.value)
print("optimal value7--x4min ", prob7.value)
print("optimal value8--x4max ", prob8.value)
print("optimal value9 --x5min ", prob9.value)
print("optimal value10--x5max ", prob10.value)

print("optimal value11--x6min ", prob11.value)
print("optimal value12--x6max ", prob12.value)

print("optimal value13--x7min ", prob13.value)
print("optimal value14--x7max ", prob14.value)

print("optimal value15--l1min ", prob15.value)
print("optimal value16--l1max ", prob16.value)

print("optimal value17--l2min ", prob17.value)
print("optimal value18--l2max", prob18.value)
print("optimal value19--l3min ", prob19.value)
print("optimal value20--l3max ", prob20.value)
print("optimal value21 ", prob21.value)
print("optimal value22 ", prob22.value)
print("optimal value23 ", prob23.value)
print("optimal value24 ", prob24.value)

print("optimal var", x1.value, l1.value)