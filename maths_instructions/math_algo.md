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

### Random search

Input Processing
Data Parsing: The parse_rows_to_arrays() function converts the input data into structured numpy arrays:
Groups strategies by actor (sector)
Extracts delta values (how each strategy affects the headline metric)
Gets private costs, weights, base payoffs, and initial behavior shares
Normalizes initial shares so each actor's strategies sum to 1
Simulation Setup
Parameters:
P_baseline: Starting value of the headline metric
P_target: Goal value we want to reach
max_epochs: Maximum number of time steps
Determines if we're moving up or down toward the target
Main Simulation Loop
For each epoch (time step):

3. Calculate Current Metric

The headline metric equals baseline plus the weighted sum of all actors' strategy effects.

4. Compute Dynamic Payoffs
   The compute_payoff() function calculates how attractive each strategy is:

Base payoff: Starting value from the data
Progress bonus: Rewards collective movement toward target (0.5 × progress)
Strategy effectiveness: Extra reward for strategies that help reach the target
Coordination bonus: Small bonus when others use the same strategy 5. Update Strategy Shares (Replicator Dynamics)
Actors gradually shift toward more profitable strategies:

fitness_diff
Higher-payoff strategies gain larger shares
Lower-payoff strategies lose share
Shares are renormalized to sum to 1 per actor 6. Check Success Condition
The simulation stops early if the target is reached:

For downward targets: P_t <= P_target
For upward targets: P_t >= P_target
Key Behavioral Dynamics
Evolutionary Pressure: Strategies that help reach the target become more attractive over time through the payoff system.

Learning Rate: Set to 0.3 for "more dynamic behavior" - actors adapt fairly quickly to changing incentives.

Stability: Minimum share values prevent any strategy from completely disappearing.

Output
Returns a SimulationResult containing:

Time series of the headline metric (P_series)
Evolution of strategy shares over time
Evolution of payoffs over time
The epoch when target was hit (if successful)
Random Search Extension
The find_optimum_random() function runs many simulations with different incentive structures:

Applies subsidies to helpful strategies (negative delta)
Applies penalties to harmful strategies (positive delta)
Searches for the lowest-budget incentive package that achieves the target
The simulation essentially models how a system of strategic actors might evolve their behavior over time in response to changing incentives and collective outcomes.

The find_optimum_random() function runs up to 5000 different simulations, each with a randomly generated incentive structure, searching for one that successfully reaches P_target.

Here's how it works:

The Search Process
For each of the 5000 trials:

Generate Random Incentives:

For strategies with delta < 0 (helpful): Apply random subsidies between 0 and subsidy_cap (0.08)
For strategies with delta > 0 (harmful): Apply random penalties between 0 and penalty_cap (0.04)
Modify Costs:

Subsidies reduce costs, penalties increase costs

Run Simulation: Execute the full evolutionary simulation with these modified costs

Check Success: If the simulation reaches P_target (i.e., result.t_hit is not None)

Track Best Solution: Keep the solution with the lowest total budget that succeeds

Early Exit Behavior
With early_exit=True (default), the search stops as soon as it finds any successful solution. This means:

It might find a solution on trial #1 and stop immediately
Or it might run all 5000 trials if no solution is found
It's looking for the first working solution, not necessarily the optimal one
Budget Optimization
The algorithm tracks the "best" solution as the one with the lowest total budget among all successful trials:

### The Best Solution Data

Looking at the code flow:

Random Search: find_optimum_random() runs up to 5000 trials and tracks the lowest-budget successful solution:

Return Best Result: The function returns (best_result, best_pi) - the simulation result with the cheapest incentive package that worked.

Generate Plots: The route calls generate_plots(result, ...) using this best_result:

What the Plots Show
The three plots display:

Metric Plot: Time series of the headline metric P_t from the best solution
Shares Plot: How strategy shares evolved over time in the best solution
Payoffs Plot: Payoff values over time in the best solution
Important Notes
If early_exit=True (default), the search stops at the first successful solution found, so the "best" solution might actually be from trial #1, #47, #2,341, etc.

If early_exit=False, it would run all 5000 trials and return the one with the lowest total budget among all successful trials.

The incentive matrix table also shows the π-values from this same best solution.

So the user sees the results from whichever trial produced the most cost-effective successful outcome, not necessarily the final trial.
