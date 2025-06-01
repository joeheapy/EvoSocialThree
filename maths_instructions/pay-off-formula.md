INPUTS:
G ← number of actor groups
K ← number of strategies per actor (usually 3)
delta[g][k] ← table of Δ-effects
cost[g][k] ← table of private costs
weight[g] ← vector of actor weights
EPSILON ← tiny positive constant (e.g. 1e-3)

OUTPUT:
payoff[g][k][t] ← 3-D array; here we fill for t = 0 (baseline)

ALGORITHM:
for each actor g in 1 … G:
for each strategy k in 1 … K:
social_gain = weight[g] \* (- delta[g][k])
raw_payoff = social_gain - cost[g][k]
payoff[g][k][0] = max(raw_payoff + EPSILON, EPSILON) # The max() guard is belt-and-braces: guarantees > 0

| Variable          | What it holds mid-loop                                                           |
| ----------------- | -------------------------------------------------------------------------------- |
| `social_gain`     | The benefit actor _g_ perceives from the strategy’s impact on the social metric. |
| `raw_payoff`      | Benefit minus private cost, _before_ applying the floor.                         |
| `payoff[g][k][0]` | Final, strictly positive pay-off stored for epoch 0.                             |
