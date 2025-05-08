from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, COIN_CMD

solver = COIN_CMD(path="/opt/homebrew/bin/cbc", msg=True)
prob = LpProblem("test", LpMinimize)
x = LpVariable("x", 0, 10)
prob += x
prob.solve(solver)
print(LpStatus[prob.status])
