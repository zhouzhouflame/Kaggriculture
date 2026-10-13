# Full-clock own-value experiment R1

Frozen before evaluation, 2026-09-26. No agent/planner changes or submission.

## Scope and data

Cover every replay state t=0..719 (Day1..30, hour0..23). Official state719
is terminal; inference there returns observed own cash exactly. Train on all
719 nonterminal states, not on a nonexistent final action. Reuse exactly the
505 training / 168 development / 168 previously-used confirmation episodes;
never open final11. Each episode and both seats remain in one split/fold.

Current legal 483 features, existing fullstage expected-shop implementation
(16 balanced, fixed exogenous scenarios); at late stages audit parity with
the original exact-known-shop features. No future shop, replay identity, seed,
offset, or opponent private inventory as input. Target is OWN terminal cash,
prediction = current own cash + learned future net gain. Both players' separate
legal-view predictions determine offline winner accuracy. This is not an
online agent reading the other player's private view.

## Frozen model comparison

Global time-conditioned CatBoost, 600 trees, depth6, learning_rate .05,
l2_leaf_reg20, RMSE, seeds93027/93028/93029. Date/hour/time-left are ordinary
features. No ranking loss and no synthetic progress examples in any fit.

Three global arms: direct terminal; half terminal plus equal-weight ALL later
nonterminal teacher values; half terminal plus ALL later nonterminal teacher
values with exponential half-life48 turns. No future-time truncation. Last
nonterminal state uses terminal directly. Terminal boundary is exact in all.

Teachers are frozen DIRECT global models from five persistent episode folds
(seed93027, identical global architecture); each training episode is annotated
only by the model excluding its whole fold. Cache all predictions once, reuse
them for both target combinations. This is ONE fitted teacher-supervision
iteration, NOT 719 recursively nested cross-fitting levels. Teacher models
use terminal labels only, so all ancestry exclusions are auditable. Repeating
fitted iterations is not silently substituted in this run.

Auxiliary stage-direct control: ten three-day heads, each500 trees depth5,
same features and training episodes, seed93027. It has more total capacity than
the global model and is a structure diagnostic, not a parameter-matched test.
Keep the existing Day30 hour16..22 direct models and pooled model as local
controls, same505 training episodes. No stage-specific tuning on confirmation.

All global settings fixed before results. No confirmation-based seed/model
selection. Training iterations and elapsed times reported, no claim of proven
optimization convergence from a fixed tree budget.

## Evaluation only

Real state metrics at ALL720 states: own-cash MAE/bias, paired winner direction,
ties/near-ties, by day/hour/stage and by seed/ensemble. Exact terminal is reported
separately, excluded from overall learned-state scores. Cluster by episode, not
state or seat. Cash baseline retained. Initial-state indistinguishability and
time/asset coverage explicitly diagnosed.

Held evaluation episodes only: seven fixed-clock own-progress slots at hour8
and hour12, offsets[-8,-4,-2,0,2,4,8]; alternate-clock triples hour6/14,
offsets[-2,0,2]. Opponent, market, shops and evaluation clock fixed. Within-day
sources only, all nonterminal. All actual training clocks are covered, so these
are NOT held-out training times (the legacy artifact key 'novel' means another
probe clock only). Evaluation games are held out. Do not train/tune on their ordering. Raw strict
order, ties, feature-identical pairs, cash-change strata and money-increment
scales reported; these are diagnostic proxies, not guaranteed legal better plans.

Cross-day probes: days2..29, hour8/16, source day[-1,0,+1]. Fix evaluation
context and replace only own farm/private state. Rebase planted/placed days,
fertilization expiry and plant lifespan steps to preserve source maturity and
remaining life; preserve -1 sentinels. Compare actual-time values separately.
These synthetic states may be unreachable and later progress is not guaranteed
better. Money reference own terminal cash*abs(offset)/720 is only an order-of-
magnitude diagnostic, not a training label or causal ground truth.

## Operations

Original data/models preserved. Resumable episode shards and cached labels.
Large remote scratch uses /root/autodl-tmp/kaggriculture_full_clock_value_r1,
not the nearly full system disk. Back up all resulting data/model/code/reports
to the independent local D:/kaggriculture/state_value_research/full_clock_value_r1
and research_backups. No credentials in artifacts. Log failures/adjustments.
