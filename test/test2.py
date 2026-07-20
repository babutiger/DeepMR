import cvxpy as cp

# Create a CVXPY problem
x = cp.Variable(2)
objective = cp.Minimize(cp.sum_squares(x))
constraints = [x >= 1]
problem = cp.Problem(objective, constraints)














# Set Gurobi parameter (parallel threads to 4)
gurobi_params = {"Threads": 120}

# Solve the problem
problem.solve(solver=cp.GUROBI, **gurobi_params)

# Output results, etc.
print("Optimal value:", problem.value)
print("Optimal solution:", x.value)
