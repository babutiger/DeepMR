import cvxpy as cp


for i in range(20, 13, -1):
    print(i)



# 假设你想要添加 i 个二进制变量，i 的值为 5
i = 5

# 使用列表推导式创建 i 个二进制变量
binary_vars = [cp.Variable(boolean=True) for _ in range(i)]

# 或者使用循环创建 i 个二进制变量
# binary_vars = []
# for _ in range(i):
#     binary_vars.append(cp.Variable(boolean=True))

# 打印结果
for idx, var in enumerate(binary_vars):
    print(f"Binary Variable {idx+1}: {var}")


import cvxpy as cp

# 假设有3个二进制变量
binary_vars = [cp.Variable(boolean=True) for _ in range(3)]

# 创建目标变量（可选，视具体问题而定）
x = cp.Variable()

# 创建优化问题
problem = cp.Problem(cp.Minimize(x), [
    # 确保至少有一个二进制变量为1
    sum(binary_vars) >= 1
])

# 解决问题
problem.solve()

# 打印结果
print("Optimal value:", problem.value)
for idx, var in enumerate(binary_vars):
    print(f"Binary Variable {idx+1}: {var.value}")
