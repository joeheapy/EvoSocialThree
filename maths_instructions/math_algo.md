### End-to-end workflow for steering the system to a quantified target

**1 — Set the objective** | Collect three numbers from the user: (a) the **baseline value** of the headline metric (e.g. 30 % poverty), (b) the **target value** (e.g. 23 %), (c) an optional **time-horizon** or “run-until-converged”. | `P_baseline`, `P_target`, `max_epochs` |

| **2 — Load the starting landscape** | Ingest or infer, for every _actor × strategy_ pair: <br>• **Δ-effect** (how the strategy moves the metric) <br>• **Private cost** to that actor <br>• **Weight** showing how much the actor cares about the social goal <br>• **Initial share** of the actor’s effort going to that strategy. | `delta[g,k]`, `cost[g,k]`, `weight[g]`, `share[g,k, epoch=0]` |
| **3 — Compute baseline pay-offs** | For each row, mix the social benefit and private cost into one positive **pay-off** value. (Keep a tiny floor so pay-offs never hit zero.) | `payoff[g,k,epoch]` |

| **4 — Run an _un-nudged_ simulation loop** | Iterate one epoch at a time: <br>1. Recalculate the headline metric from current shares. <br>2. Update each actor’s strategy shares using the replicator rule (“strategies with higher pay-off gain share”). <br>3. Log the metric, the shares, and every pay-off. | Time-series of `P`, `share`, `payoff` |

| **5 — Check for success or stagnation** | • **If the target is met** → record this mix as an _optimum_, note the epoch number, jump to Stage 8. <br>• **If the loop stalls** (metric stops improving, or max_epochs reached) → flag “needs incentives” and move on. | `optima[]` list, or a “needs-incentive” flag |

| **6 — Design incentives / penalties** | Search for the smallest set of subsidies, taxes, or rules that will lift the pay-off of helpful strategies (or lower that of harmful ones) just enough to make progress possible. Store these adjustments so they can be tweaked later. | `incentive[g,k]` (often called
**π-matrix**) |

| **7 — Re-simulate with incentives** | Reset to the original shares, apply the π-matrix to the cost figures, recompute pay-offs, and run the loop again. Repeat Stages 5–7 until at least one optimum is reached or the budget for incentives is exhausted. | Updated time-series; possible new `optima[]` entries |

| **8 — Catalogue all optima** | Whenever the metric first drops to or below the target, save: <br>• the epoch number, <br>• the full roster of strategy shares, <br>• pay-offs, weights, and incentives in force, <br>• the running total cost of incentives. Continue the search if you want _all_ cost-equivalent optima. | Append to `optima[]`  
 |
| **9 — Generate the report pack** | For each optimum found, output: <br>• Number of epochs taken <br>• Final metric value <br>• Strategy mix per actor <br>• Pay-off table <br>• Incentive spend <br>Provide trajectory charts and a downloadable CSV/JSON for auditing. | Rendered tables, plots, `optima_summary.csv` |

| **10 — Optional stress-tests** | Re-run the entire loop with ± 10 % noise on Δ-effects, costs, or weights to see how fragile each optimum is. Flag any that collapse under modest uncertainty. | Sensitivity dashboards |
