Step-by-step prompt sequence for steering a social system towards a quantified target

Context reminder: Actors and strategies are already listed (see attached tables).
Target example: “Reduce the number of families kept in B&B accommodation beyond six weeks in London from 1 287 to 0 within 3 years.”

---

### 1 Compile the starting data

- Gather, for every actor and every strategy, three things:

  1. An estimate of how much that strategy helps or harms the headline target (for example, the number of families still in emergency accommodation).
  2. An estimate of what the strategy costs the actor (money, staff time, political risk, and so on).
  3. A rough weight that says how much the actor actually cares about the headline target compared with its own cost.

- Store those three sets of numbers in a tidy, machine-readable table.

### 2 Turn those numbers into pay-offs

- For each row in the table, calculate a single figure that mixes “doing good for the target” with “paying the private cost”.
- Keep a tiny positive floor value so none of the pay-offs accidentally drop to zero or become negative; this keeps later calculations safe.

### 3 Record the situation today

- Write down, for every actor, what share of their effort is currently spent on each strategy.
- Note the current value of the headline target, such as “1 287 families kept in B\&Bs for more than six weeks”.

### 4 Describe how behaviour shifts over time

- Choose one clear rule that says:
  “If a strategy is giving its actor a bigger pay-off than the average of that actor’s other strategies, the actor will lean into it next time-step, and vice versa.”
- State that rule in ordinary language that can later be translated into code for each actor separately.

### 5 Run a first, un-incentivised simulation

- Feed the starting shares, the pay-offs, and the behaviour-shift rule into a loop that steps forward once per chosen period (for example, once per quarter).
- Let the loop run for as many periods as the policy timetable demands (three years means twelve quarters).
- Track how the headline target moves. If the target is met at or before the deadline, save that trajectory; if not, make a note of the shortfall.

### 6 Plan nudges and penalties when the first run fails

- Introduce a second table that lists possible subsidies, tax breaks, regulations, or fines that would make the helpful strategies cheaper or the unhelpful ones dearer.
- Search for the smallest bundle of nudges and penalties that — when added to the cost figures — allows the simulation to hit the target on time.

### 7 Re-run the simulation with those policy tweaks

- Swap the original cost numbers for the tweaked ones.
- Repeat the time-stepping loop.
- Confirm that the headline target is now achieved by the deadline and that all the strategy shares stay between zero and one and always add up to one for each actor.

### 8 Explore alternative solutions

- Relax one design choice at a time — perhaps the size of the incentive budget, or which actors are allowed to receive subsidies — and re-run the search.
- Collect any additional combinations that also meet the target, cost the same, or cost less, and sort them for decision-makers.

### 9 Stress-test and document the whole set-up

- Vary the key input guesses (the effect estimates, the cost figures, and the actor weightings) to see how sensitive the result is.
- Keep a clear log of every assumption, data source, and tweak so that the final code can reproduce the whole process end-to-end for audit and update.

---

#### Recap & memory hooks

1. **Three source tables** – effect size, private cost, actor weight.
2. **One behaviour rule** – better pay-off means bigger share next round.
3. **First dry run** – find out whether the system fixes itself without help.
4. **Smallest nudge hunt** – add only the incentives needed to land on target.
5. **Stress-test record-keeping** – make the model reproducible, transparent, and easy to rerun.
