# Hash routing in mixture-of-experts language models

We are using this code to write a paper for ICML 2027: **"Hash Routing: When Should Each Token in the Vocabulary Have Its Own Fixed Expert?"** The draft is [PAPER.md](PAPER.md) ([PAPER.html](PAPER.html) renders the figures). The code is under the MIT licence.

**Main rule.** A fixed token-to-expert table cannot split a token, so its busiest expert carries at least max(1, p E/k) times the average load, where p is the most common token's share of text (about 1/26 for English web text), E the number of experts and k the experts per token. A table balanced by token frequency reaches this limit in every design tested, and only E/k matters. In loss, the table never trained worse than a learned router in our tiny models.

**Reproduce.**

| File | What it does | Time |
|---|---|---|
| `rule.py` | Exact load of random and balanced tables for 14 designs and 2 tokenizers, writes `rule.json` | 10 s |
| `hr.py` | MoE test bed (learned vs table routing, capacity, churn, forced moves); every flag in `--help`; appends to `results.jsonl` | 3-30 s per run |
| `load.py`, `curve.py`, `sweep.py`, `table.py`, `fit.py` | Load checks, gap along training, sweeps, tables and fits for the paper | seconds |

Data (not in the repo, `data/`): `counts.npy` (GPT-2 token counts of FineWeb), `facts.json` (planted ParaRel sentences), and FineWeb GPT-2 token shards `fineweb_train_000001.bin` and `fineweb_val_000000.bin` from the modded-nanogpt cached FineWeb10B set.

---

# Lab notebook

## Hash routing in the first layers of a mixture-of-experts LLM

**Question.** DeepSeek-V4 picks the experts in its first 3 layers from the token ID alone (a fixed table, no learned router), and a learned router in all later layers. Why does that work, how many layers can do it, and what do those experts store?

**Setup.** `hr.py`: the plain transformer of `llm-value-embeddings/ve.py` (RMSNorm, RoPE, QK-norm, Muon on matrices, Adam on the rest), with every feed-forward layer replaced by a DeepSeek-style MoE: 1 shared expert plus top-2 of 8 routed experts, active width equal to a dense FFN, sqrt(softplus) router scores, aux-loss-free balancing. Hash layers use a random fixed token-ID → 2-expert table; the router still sets the 2 weights, as in V4. Data: FineWeb plus planted ParaRel facts written 2, 8 or 32 times, from the memory paper. Data files are symlinks into `llm-value-embeddings/data` and `muon/ngram-hints/data`.

## First run (2026-11-12): width 48, 4 layers, 120 steps, one run each

| Arm | Web loss | Stored facts, 32 copies (loss) | Relation gap, 32 copies (higher is better) | Learned layers: share of a token's uses sent to its usual experts (chance) |
|---|---|---|---|---|
| Dense FFN | 5.045 | 3.060 | 2.48 | - |
| MoE, all layers learned | 5.041 | 3.016 | 3.16 | .56 to .71 (.13 to .19) |
| Hash layer 0 | 5.039 | 2.945 | 3.52 | .64 to .76 |
| Hash layers 0, 1 | 5.026 | 2.881 | 3.71 | .64 to .66 |
| Hash last layer only | 5.030 | 2.950 | 3.25 | .63 to .69 |
| Hash all 4 layers | 5.020 | 2.841 | 3.91 | - |

**Problems with this run.** The Mac was busy, so 5 of the 6 runs took 10 to 22 s of training, over the 10 s limit. Rerunning learned vs hash-all with two other model initializations on a quiet Mac gave the same order every time (hash-all lower by 0.021 to 0.032), so the spread between runs is about 0.01.

## Sweep (2026-11-12): hash all layers vs learned all layers, one run per setting

| Setting | Steps | Web loss, learned | Web loss, hash all | Difference | Stored facts 32 (learned / hash) | Relation gap 32 (learned / hash) | Learned share per layer (chance) |
|---|---|---|---|---|---|---|---|
| width 32 | 120 | 5.115 | 5.082 | -0.033 | 3.72 / 3.43 | 2.48 / 3.33 | .60 to .70 (.15 to .19) |
| 2 layers | 120 | 5.059 | 5.030 | -0.029 | 2.91 / 2.83 | 3.51 / 4.07 | .63, .69 (.17, .24) |
| 16 experts | 120 | 5.044 | 5.003 | -0.041 | 3.21 / 2.84 | 2.83 / 3.74 | .50 to .56 (.11 to .15) |
| 4 experts, top-1 | 120 | 5.048 | 5.026 | -0.022 | 3.13 / 2.87 | 2.76 / 3.44 | .70 to .84 (.32 to .34) |
| width 96 | 80 | 5.134 | 5.113 | -0.021 | 3.07 / 2.59 | 2.39 / 3.21 | .61 to .74 (.19 to .29) |
| 6 layers | 100 | 5.176 | 5.119 | -0.057 | 3.94 / 3.25 | 1.93 / 3.00 | .50 to .72 (.14 to .24) |

## Over training (width 32, 2 layers, 400 steps; runs took 12 s, over the limit)

| Step | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 |
|---|---|---|---|---|---|---|---|---|
| Web loss difference (hash minus learned) | -0.056 | -0.027 | -0.023 | -0.002 | -0.005 | +0.004 | +0.007 | +0.003 |
| Learned layer 0 share | .73 | .72 | .66 | .85 | .82 | .83 | .87 | .88 |
| Learned layer 1 share | .81 | .87 | .60 | .62 | .64 | .57 | .64 | .63 |

At step 400 the hashed model still stores facts better (2.56 vs 2.81) with a larger relation gap (3.53 vs 3.23).

## What it says

1. Early in training a fixed hash beats a learned router in every setting tried, because a learned router has not learned anything beyond token ID yet and wastes steps finding a stable assignment.
2. The web-text advantage is gone by step 200; the facts advantage remains at step 400.
3. With training, the first layer's learned router becomes more of a hash (73% to 88% of a token's uses go to its usual experts), while the second layer moves away (81% to 63%). This matches DeepSeek-V4 hashing only its first layers.

## The rule (2026-11-12, runs at batch 16, up to 400 steps)

**Rule: a fixed token-ID hash routes as well as the best learned router, and better whenever a token's output rides on few routing choices. Hash routing gives nearly the same loss whatever the MoE design; the learned router gets worse the more of each token's feed-forward output depends on one routing decision (no shared expert, top-1).**

Width 32, 2 layers, 400 steps, all layers hashed vs all learned:

| MoE design | Weight on one routing choice | Learned | Hash | Hash minus learned |
|---|---|---|---|---|
| Shared expert, top-2 of 8 | 0.25 | 4.850 | 4.869 | +0.019 |
| No shared expert, top-2 of 8 | 0.5 | 4.865 | 4.860 | -0.005 |
| Shared expert, top-1 of 8 | 0.5 | 4.879 | 4.857 | -0.022 |
| No shared expert, top-1 of 8 | 1.0 | 4.914 | 4.858 | -0.056 |

The same order at other sizes (250 steps):

| Size | Shared, top-2 | No shared, top-1 |
|---|---|---|
| Width 64, 2 layers | -0.008 | -0.025 |
| Width 32, 4 layers | -0.025 | -0.069 |

Checks on the worst case for learned routing (no shared expert, top-1):

| Change | Learned | Hash |
|---|---|---|
| Balancing step 0.01 (default) | 4.914 | 4.858 |
| Balancing step 0.001 (DeepSeek-V4) | 4.911 | |
| No balancing | 4.897 | |

Why: the learned router keeps moving tokens between experts (15 to 90% of positions change expert every 50 steps, falling only near the end), so expert work is thrown away. A shared expert or a second routed expert absorbs that; a fixed hash never moves.

**When to use it.** Always safe at these sizes; most useful in top-1 or no-shared-expert designs, where it replaces a router that costs loss.

The earlier per-layer guess (hash layer 0 free, layer 1 costly) did not hold: at 400 steps with a shared expert, hashing layer 0, layer 1 or both changed loss by +0.010, +0.006, +0.019, all within noise.

**When not to use it: not found yet.** Hashing never lost by more than 0.019 (within about 2x the run-to-run spread). The expected failure case, where the same word needs different processing in different contexts, did not appear: with half the training rows reversed, the learned router does route by context (token share falls from .87/.61 to .55/.53), yet hash still ties (5.181 vs 5.187).

**Open.** All models are 0.3 to 1.7M parameters and at most 400 steps. In the no-shared top-1 design the hash lead shrank from 0.10 at step 100 to 0.056 at step 400, so it may close with long training. DeepSeek-V4 hashing only its first 3 layers suggests later layers pay a cost at scale that these runs cannot reach. The 400-step runs took 6 to 15 s depending on the Mac's load; several went over 10 s.

## Why: routing churn (2026-11-12, factor sweep)

Base: width 32, 2 layers, no shared expert, top-1 of 8, 200 steps, batch 16 (`sweep.py`, `table.py`). "Churn" = share of validation positions whose expert changed since the snapshot 100 steps earlier, averaged over training.

**Factors (hash minus learned web loss; the learned router's churn):**

| Factor | Value | Hash minus learned | Learned churn |
|---|---|---|---|
| Base | | -0.079 | 0.54 |
| Vocabulary | 1000 | -0.068 | 0.56 |
| Vocabulary | 500 (each word seen ~8x more often) | -0.060 | 0.61 |
| Experts | 4 | -0.030 | 0.40 |
| Experts | 16 | -0.104 | 0.65 |
| Sequence length | 16 | -0.089 | 0.59 |
| Sequence length | 96 | -0.092 | 0.56 |
| Learning rate | 2x | -0.093 | 0.62 |
| Learning rate | 1/4 | -0.082 | 0.44 |
| Cooldown | 80% of training | -0.088 | 0.43 |
| Router learning rate | 0.1x, 0 (frozen), 3x | -0.074, -0.081, -0.077 | 0.52, 0.53, 0.54 |
| Router reads only the token embedding | | -0.062 | 0.32 |
| Hash table balanced by token frequency | | +0.012 (vs random table) | 0 |

The fixed table won in every one. The router's own learning rate does not matter, even a frozen router: churn comes from the router's input (the hidden state) changing as the rest of the model trains. A router that reads only the token embedding routes purely by word (share 1.00) but its choices still move (churn 0.32), and it loses almost as much. So the cost is not context, it is the word-to-expert assignment moving.

**Causal tests.**

| Arm | Churn | Web loss |
|---|---|---|
| Fixed table | 0 | 5.065 |
| Table, 10% of words re-assigned every 25 steps | 0.20 | 5.110 |
| Table, 50% re-assigned every 25 steps | 0.81 | 5.262 |
| Learned router | 0.54 | 5.144 |
| Learned 50 steps, then frozen into a table | 0 after step 50 | 5.091 |
| Learned 100 steps, then frozen | 0 after step 100 | 5.101 |
| Shared expert, top-2: learned | 0.75 | 5.070 |
| Shared expert, top-2: fixed table | 0 | 5.068 |
| Shared expert, top-2: table, 50% re-assigned | 0.87 | 5.097 |

Moving a fixed table on purpose reproduces the loss, in proportion to how much it moves. Freezing a learned router helps, but not past a random table frozen from step 0: what the router learned was worth nothing here, only stability was. With a shared expert and top-2, the same forced churn costs 7x less (0.029 vs 0.197), which is why that design ties.

## The rule

**Loss lost to routing ≈ (how much the word-to-expert assignment moves during training) × (how much of a word's output rides on one routing choice). A fixed hash has zero movement, so it wins wherever the second factor is large and ties where it is small.**

- When to use a fixed table: when each word goes to few experts and there is no shared expert; when a learned router's assignments are still moving.
- How to use it: a random table is enough; balancing it by word frequency did not help. Freezing a learned router's choices into a table also works, but here it was no better than random.
- When not to use it: when the learned assignment carries value once stable. Not seen at this size: frozen learned tables never beat a random one.

**What this predicts for long training (not run).** Churn falls late in training (to 0.15 to 0.25 per 50 steps by step 400 in the earlier run), so the learned router's ongoing cost should shrink, which matches the gap shrinking over 400 steps. Whether it then overtakes depends on whether its stable assignment beats a random one, which these runs say it does not at this size. Lower learning rate and long cooldown, two stand-ins for late training, did not reduce churn much (0.43 to 0.44) and did not shrink the gap. At scale, later layers may gain from context routing, which would explain DeepSeek-V4 hashing only its first 3 layers.

**Run limits.** The Mac was loaded (load average 6 to 11) by other sessions; 9 of the ~35 runs took 10.4 to 24.6 s, over the 10 s limit, the worst being 16 experts (all experts are computed densely).

## Literature check (2026-11-12, web search, abstracts and summaries; full texts not read)

**Already known**

| Paper | What it already shows | Overlap with us |
|---|---|---|
| Hash Layers, Roller et al. 2021 ([2106.04426](https://arxiv.org/abs/2106.04426)) | Top-1 hash beats Switch; the lead grows from 16 to 128 experts; random hash about equal to balanced (23.22 vs 23.16 PPL). Says in words that learned routing is hard because "membership for each expert is changing while it is trying to learn the mapping". Hash works worst in layer 1 | Our expert-count trend, random-vs-balanced result and the churn idea are all here, the last one as an untested remark |
| StableMoE, Dai et al. ACL 2022 ([2204.08396](https://arxiv.org/abs/2204.08396)) | Named "routing fluctuation": 40.9% of tokens still change expert after 20% of training. Freezing a distilled word-embedding router after 10% of training lowers PPL; freezing later is worse. Frozen learned router beats Hash Layer (18.93 vs 19.25 PPL) at 124M backbone, top-1, 60K steps. No causal test, no random frozen baseline, no shared experts | Mechanism named and measured; the fix (freeze assignments) is theirs |
| OpenMoE, Xue et al. ICML 2024 ([2402.01739](https://arxiv.org/abs/2402.01739)) | Learned routing is mostly by token ID and is fixed early in pretraining | Matches our token share .5 to .9 |
| Fan, Messmer, Jaggi 2024 ([2402.13089](https://arxiv.org/abs/2402.13089)) | Frozen random router about equal to a learned one | Matches our frozen-router loss |
| Clark et al. ICML 2022 ([2202.01169](https://arxiv.org/abs/2202.01169)) | Hash worse than S-BASE for most sizes up to hundreds of billions of parameters, with a large overhead at small size | Opposite direction at scale |
| Ternovtsii & Bilak 2026 ([2604.14419](https://arxiv.org/abs/2604.14419)) | Routing topology barely matters at 76 to 84M; hash and random-fixed cost 1.1 to 2.2 PPL | Mixed |
| Dikkala et al. EMNLP 2023 | Learned routing helps when data has cluster structure | A "when not to hash" condition |

DeepSeek-V4 gives no reason for hashing its first 3 layers (sec. 2.1 only says it replaces the dense FFNs there).

**Not found in any paper (possible contributions)**

1. Causal dose test: re-assigning a share of a fixed table's words reproduces the learned router's loss, in proportion to the churn. StableMoE's evidence is only correlational.
2. The shared-expert and top-k interaction: the same churn costs 7x less with a shared expert and top-2. This explains when hashing is free (DeepSeek-style designs) and when it wins (top-1, no shared expert). The one related result found ([2604.23036](https://arxiv.org/abs/2604.23036)) gives a variance argument for always-on experts, with no churn measurement.
3. Where churn comes from: a frozen router churns as much as a learned one (0.53 vs 0.54), and an embedding-only router still churns (0.32). Churn comes from the router's input drifting, so slowing or freezing the router weights does not help; only freezing the assignment does.

**Conflicts to resolve before claiming anything**

- StableMoE's frozen learned router beats Hash at 124M and 60K steps; ours did not beat a random table. A learned assignment probably gains value with size and training, so "a random table is enough" is a small-scale result only.
- Clark et al. find hash worse at scale. Their learned router (S-BASE) is balanced by assignment and may churn less; churn was not measured.
- Roller finds hash works worst in layer 1, while DeepSeek-V4 hashes layers 1 to 3. Our per-layer test was within noise.

**Paper angle.** "Routing churn times routing weight": a measured, causal account of why fixed routing matches learned routing, and of when a shared expert makes the choice irrelevant. The claims must stay inside these conflicts.

## Rule across 8 MoE designs (2026-11-12, `fit.py`)

Width 32, 2 layers, 200 steps, batch 16, 8 experts unless noted. Per design: learned router, fixed random table, and the fixed table with 50% of words re-assigned every 25 steps (forced churn). w = share of the feed-forward width one routed expert holds (1/k without a shared expert, 1/(2k) with one).

| Design | w | Learned | Fixed table | Table minus learned | Learned churn | Forced-churn cost per unit churn |
|---|---|---|---|---|---|---|
| No shared, top-1, 4 experts | 1 | 5.116 | 5.086 | -0.030 | 0.40 | 0.116 |
| No shared, top-1, 8 experts | 1 | 5.144 | 5.065 | -0.079 | 0.54 | 0.243 |
| No shared, top-1, 16 experts | 1 | 5.185 | 5.081 | -0.104 | 0.65 | 0.264 |
| No shared, top-2 | 1/2 | 5.095 | 5.073 | -0.023 | 0.74 | 0.103 |
| No shared, top-4 | 1/4 | 5.086 | 5.075 | -0.012 | 0.77 | 0.044 |
| Shared, top-1 | 1/2 | 5.106 | 5.059 | -0.047 | 0.49 | 0.067 |
| Shared, top-2 | 1/4 | 5.070 | 5.068 | -0.003 | 0.75 | 0.033 |
| Shared, top-4 | 1/8 | 5.080 | 5.073 | -0.007 | 0.72 | 0.024 |

1. The fixed table scores 5.059 to 5.086 in every design; the learned router ranges from 5.070 to 5.185 and is worse in all 8.
2. The cost of one unit of churn follows w (correlation 0.88 over the 8 designs; fit 0.22 × w), and also grows with the number of experts (0.116, 0.243, 0.264 for 4, 8, 16). More experts means each expert has seen a narrower set of words, so a word moved to a new expert lands somewhere that has not learned it.
3. Forced-churn cost per unit × learned churn predicts the learned router's loss gap with correlation 0.92, but about 1.6x too large: a learned router's moves are cheaper than random moves (likely because it moves words the new expert has partly seen). Shared top-1 is the one design where the gap (0.047) is larger than predicted (0.033).

**Rule, refined.** The loss a learned router loses to a fixed table ≈ how often it switches a word's expert × how much of the word's feed-forward output that one expert carries × a factor that grows with the number of experts. Top-k and a shared expert both shrink the middle term; with a shared expert and top-2 or more, the gap is under 0.01.

All 16 runs took 2.9 to 6.7 s.

## Long training imitated with tiny models (2026-11-12)

Question: does the fixed table's lead survive when each weight sees as many tokens as in a large model? Shrink the model (width 16 or 8, vocabulary 500 so the embedding does not dominate) and train it to 37-124 tokens per weight in 14-26 s. Each token is still read once. `curve.py L H` prints the gap along training.

| Model | Design | Steps x batch | Tokens per weight | Table minus learned at end |
|---|---|---|---|---|
| width 16 | no shared, top-1 | 300 x 16 | 4.7 | -0.023 |
| width 16 | no shared, top-1 | 2400 x 16 | 37 | -0.010 |
| width 16 | no shared, top-1 | 300 x 128 | 37 | -0.013 |
| width 16 | shared, top-2 | 2400 x 16 | 66 | -0.009 (-0.021 to +0.002 along the way) |
| width 8 | no shared, top-1 | 2400 x 16 | 124 | +0.002 (+0.004 to +0.007 from step 1200 to 1800) |

Takeaway: the table's lead shrinks as tokens per weight grows and is gone by about 100 tokens per weight; the learned router never wins by more than noise. Tokens seen matter, not steps (big batch with few steps gives the same gap).

Why: learned churn stays 0.3-0.5 even at 124 tokens per weight, so the learned router still moves words. Random moves still cost: a width-16 table forced to move 50% of words every 25 steps loses 0.12 at step 300 and a steady 0.05-0.06 from step 600 to 2400 (forced churn about 0.9). So the learned router's own moves become nearly free late in training, probably because it moves words to experts that already handle similar words. Earlier short runs already showed learned moves costing about 1.6x less than random ones.

Consequence for the paper: "the table matches or beats the learned router" holds at every length tested, but "often better" holds only early in training or in short runs. At large-model token counts the two tie in our tiny models; whether a learned router pulls ahead at real scale (as StableMoE and Clark 2022 suggest) is not settled by these runs.

## Large-scale factors (2026-11-12, second round)

**Unknown-word fix.** At vocabulary 500, 42% of text is the unknown token (20% at 4,000; 0 in a real 50K vocabulary, where the top token is 3.8%). The table sent all of it to one expert. New flag `--unk router`: in hashed layers the unknown token uses the learned router, so the table only covers real words. This raised the short-run table lead (-0.023 -> -0.039) and barely moved the long runs.

| Tiny model, long training, `--unk router` | Tokens per weight | Table minus learned at end |
|---|---|---|
| width 16, no shared, top-1 | 37 | -0.006 |
| width 8, no shared, top-1 | 124 | -0.001 |
| width 16, 64 experts, shared + top-4 | 35 | -0.001 (short run at 4.4: -0.014) |
| width 8, 4 layers, hash layer 0 only | 124 | -0.004 vs all learned |
| width 8, 4 layers, hash layer 3 only | 124 | -0.001 vs all learned |

Takeaway: at large-model token counts the table and the learned router tie in every design and layer tested, including 64 small experts where the learned router moves words very often (churn 0.6-0.99) and pays nothing because each expert carries 1/8 of the output. The learned first layer becomes a token table by itself (93% of a word's uses go to its usual expert; last layer 60%), and hashing either layer costs nothing.

**Load balance at real scale (exact, `load.py`).** A fixed table cannot split a token, so the experts of the most common token (3.8% of FineWeb with GPT-2 tokens) carry at least 3.8%/k of all load against a fair 1/E. Busiest expert / average:

| Experts, top-k | Random table | Frequency-balanced table | Floor = top share x E / k |
|---|---|---|---|
| 8, top-1 | 1.20 | 1.00 | 1.00 |
| 64, top-4 | 2.09 | 1.00 | 1.00 |
| 256, top-8 | 3.07 | 1.22 | 1.22 |
| 256, top-6 | 3.98 | 1.63 | 1.63 |
| 384, top-6 (DeepSeek-V4 size) | 3.75 | 2.44 | 2.44 |

Rule: once E/k exceeds 1 / (top token share), about 26 here, no fixed table can be balanced; at DeepSeek-V4's size the busiest expert gets at least 2.4x the average. A learned router with bias balancing can split common tokens. DeepSeek-V4's tokenizer and data differ, so its exact number differs.

## Capacity cap: does the table's imbalance cost loss? (2026-11-12, third round)

New flag `--cap c`: each expert takes at most c x (tokens x k / E) tokens per batch; later tokens are dropped (weight 0, residual passes). Logged per layer: `route.i.drop` (share dropped) and `route.i.trainload` (busiest / average per batch). Setting: 64 experts, shared + top-1 (64 experts per chosen expert, the same ratio as DeepSeek-V4's 384 with 6 chosen), width 8, vocabulary 16,000 (6.7% unknown, `--unk router`), batch 64, 200 steps (short training, about 2 tokens per weight).

| Arm | Web loss | Dropped | Busiest / average |
|---|---|---|---|
| Learned, no cap | 6.650 | 0 | 3.27 |
| Learned, cap 1.25 | 6.651 | 13.5% | 3.36 |
| Table, no cap | 6.526 | 0 | 3.77 |
| Table, cap 1.25 | 6.570 | 16.5% | 3.77 |

Takeaway: the table's drops cost 0.044 loss, the learned router's drops almost nothing (0.001); the cap cuts the table's short-training lead from 0.124 to 0.081. The table probably pays more because it always drops the same frequent tokens' expert, while learned drops land on different tokens. The learned router's 3.27 imbalance is a short-run artifact (its bias balancing has 200 steps); at scale learned routers balance closely.

Exact drop share for a very large batch (`load.py`, no sampling noise):

| Experts, top-k | Table | Dropped at cap 1.0 | cap 1.25 | cap 2.0 |
|---|---|---|---|---|
| 64, top-1 | random | 22.9% | 17.4% | 6.8% |
| 64, top-1 | balanced | 8.0% | 5.4% | 1.5% |
| 64, top-4 | random | 15.0% | 7.9% | 0.2% |
| 64, top-4 | balanced | 0 | 0 | 0 |
| 384, top-6 | random | 22.2% | 16.7% | 7.2% |
| 384, top-6 | balanced | 8.0% | 5.4% | 1.5% |

The training run's 16.5% matches the predicted 17.4%. Rule: the imbalance and drop share depend on E/k (experts per chosen expert), not on E alone: 64/1 and 384/6 give nearly the same numbers, so a tiny 64-expert top-1 model reproduces DeepSeek-V4's table imbalance. At that ratio a random table drops about 17% at cap 1.25, a frequency-balanced table 5%. With no cap (dropless training), the same imbalance costs step time instead: the busiest expert does 2.4x (balanced) to 3.75x (random) the average work.

**Frequency-balanced table under the cap** (`--hashtype balanced`, same setting):

| Arm | Web loss | Dropped | Busiest / average |
|---|---|---|---|
| Random table, no cap / cap 1.25 | 6.526 / 6.570 | 0 / 16.5% | 3.77 |
| Balanced table, no cap / cap 1.25 | 6.529 / 6.556 | 0 / 7.7% | 2.72 |
| Learned, no cap / cap 1.25 | 6.650 / 6.651 | 0 / 13.5% | 3.27 |

Takeaway: balancing halves the drops (prediction 5.4% for a very large batch; the rest is batch noise at 47 tokens per expert) and the drop cost falls with them (0.044 -> 0.027, about 0.3 loss per 100% dropped). Without a cap it changes nothing. So at a high experts-per-choice ratio a fixed table should be frequency-balanced; it still cannot get below the floor set by the most common token.

## One claim, three focused tests (2026-11-12, fourth round)

Claim under test: **in final loss a fixed token-to-expert table equals a learned router; its only real cost is load imbalance, and that cost is set by token frequencies and experts per chosen expert (E/k).**

New flag `--unk skip`: in every MoE layer the unknown token uses only the shared expert and is left out of load, caps and balancing (both arms), so load numbers cover real words only.

**Test 1: longer than before.** Width 8, 8 experts top-1, vocabulary 500, `--unk router`, batch 128 x 600 steps = 248 tokens per weight (DeepSeek-V3: about 22 tokens per total weight).

| Step | 100 | 200 | 300 | 400 | 500 | 600 |
|---|---|---|---|---|---|---|
| Table minus learned | -0.064 | -0.022 | -0.013 | -0.005 | -0.002 | -0.001 |

The gap shrinks to zero and does not cross: the learned router does not pull ahead at 2x the previous longest run.

**Test 2: DeepSeek-like design.** Shared expert + 192 experts top-3 (E/k = 64, the same as DeepSeek-V4's 384 / 6), width 8, vocabulary 500, `--unk skip`, 81 tokens per weight.

| Arm | Loss at end | Gap vs table | Busiest / average (layer 0 / 1) |
|---|---|---|---|
| Table | 2.948 | | 5.9 / 8.8 |
| Learned, bias step 0.01 | 2.965 | +0.017 | 6.1 / 11.5 |
| Learned, bias step 0.05 | 2.972 | +0.024 | 8.1 / 10.8 |

Gap along training: -0.048, -0.028, -0.018, -0.017. With many small experts the table keeps a lead of about 0.02 at 81 tokens per weight, and the learned router balances no better than the table (churn 0.84-0.99). A faster bias step does not fix its balance. Tiny models cannot show the balance a real learned router reaches, so the "learned routers balance better" half of the claim rests on published large models, not on these runs.

**Test 3: cap at two ratios, longer training.** Shared + top-1, 64 or 16 experts, width 8, vocabulary 4,000, `--unk skip`, batch 64 x 400 steps.

| Experts | Arm | No cap | Cap 1.25 | Cap cost | Dropped | Predicted drop | Busiest / avg (measured / predicted) |
|---|---|---|---|---|---|---|---|
| 64 | Table | 4.835 | 4.856 | 0.021 | 20.6% | 20.1% | 3.86 / 3.63 (L0), 4.09 / 3.99 (L1) |
| 64 | Learned | 4.875 | 4.894 | 0.019 | 15.2% | | 3.1 / 3.7 |
| 16 | Table | 4.860 | 4.866 | 0.006 | 9.8% | 9.7% | 2.34 / 2.33 (L0), 1.64 / 1.55 (L1) |
| 16 | Learned | 4.889 | 4.888 | -0.001 | 4.0% | | 1.5 / 1.8 |

Predicted values come from token frequencies and the exact tables used, with no training. They match the measured drop to within 0.5 points and the busiest load to within 0.2.

Takeaways:
1. The table's load and drop share are predicted exactly from token frequencies and E/k before training.
2. Drops cost about 0.06-0.13 loss per 100% dropped once training is longer, the same for both routers (the short 200-step run gave 0.27 for the table and near 0 for learned; that difference does not survive longer training).
3. Even with the cap, the table stays ahead of the learned router in these tiny models (64 experts: -0.038; 16 experts: -0.022), because the learned router is no better balanced here.
4. So the claim's first half holds (no loss lost to the table at any length; tie at 248 tokens per weight with few experts, small table lead with many experts). Its second half holds for the table side (cost = drop share or step time, predicted from E/k). The comparison side, that a learned router balances better, is not reproduced by tiny models.

## The load rule (fifth round, 2026-11-12)

**Rule.** A fixed table cannot split a token, so the busiest expert carries at least max(1, p E/k) times the average, where p is the most common token's share (3.82% GPT-2, 3.78% cl100k; 1/p is about 26). A frequency-balanced table reaches it; only E/k matters. Under capacity c, a balanced table overflows by the sum of max(0, p_i - c k/E).

**Exact check (`rule.py`, 10 s, no training, 2 tokenizers x 14 designs, `rule.json`).**

| E/k | Limit | Balanced | Random | Overflow at 1.25: formula / balanced / random |
|---|---|---|---|---|
| 16 | 1.00 | 1.00 | 1.73-2.30 | 0 / 0 / 3.0-7.7% |
| 32 | 1.22 | 1.27 | 3.05-3.12 | 0 / 0.1% / 7.7-11.4% |
| 64 | 2.44 | 2.49 | 3.1-6.3 | 5.4% / 5.8% / 14.1-18.3% |
| 128 | 4.88 | 4.93 | 5.7-6.3 | 12.9% / 13.3% / 19.7-21.8% |
| 256 | 9.77 | 9.82 | 10.8 | 19.4% / 19.8% / 25.4-26.9% |

Balanced sits 0.05 above the limit everywhere; 64/1, 128/2, 256/4 and 384/6 give the same balanced load.

**Learned routers hit the same limit** (64 experts top-1 shared, vocab 4,000, limit 3.07, last quarter of training): 2 layers 3.68 / 3.15 (token share 0.94 / 0.65); 4 layers 3.13 / 3.27 / 3.59 / 3.18 (token share 0.80 / 0.80 / 0.74 / 0.74).

**Capacity sweep, 64 experts.** Random table: cap 2.0 / 1.25 / 1.0 drop 9.8 / 20.6 / 26.8%, loss 4.853 / 4.856 / 4.855 (no cap 4.835). Balanced table at 1.25: drops 10.5%, loss 4.833 -> 4.835.

**Noise.** Three repeats at 64 experts: table 4.835 / 4.847 / 4.840, learned 4.875 / 4.887 / 4.872; gap -0.037, spread 0.012 / 0.015.

Takeaway: the paper is rewritten around the load rule (PAPER.md, claims 1-3); the loss claims (table never worse, churn explains the early lead) are claims 4-6.

## Which layers should use a table (sixth round, 2026-11-12)

Many 5-30 s runs. New flags: `--ctx` (after training, replace one learned layer's routing by each word's most common expert set, and by a random table, at test time) and `--split P` (every word written as two pieces, so no piece names a word).

**Early gain sits in layer 0** (hashed minus learned, 200 steps, width 32):

| Model | Hash L0 only | Hash L1 only | Later layer only | Hash all | Random-table cost L0 / later |
|---|---|---|---|---|---|
| 4 layers, top-1 | -0.098 | -0.003 | -0.012, -0.010 | -0.110 | 0.169 / 0.004-0.018 |
| 6 layers, top-1 | -0.111 | -0.026 | -0.013, -0.002 | -0.125 | 0.151 / 0.002-0.011 |
| 4 layers, shared top-2 | -0.013 | +0.003 | 0.000 | -0.016 | 0.041 / 0.003-0.005 |
| 4 layers, split words | -0.026 | -0.010 | -0.008 | -0.028 | 0.342 / 0.004-0.007 |

Churn is the same in every layer (0.58-0.74), so the gain follows how much the layer's expert choice matters.

**A learned router's context is worth nothing in short runs**: forcing a per-word table at test time changed loss by -0.017 to -0.001 in 10 settings (depth, width 64, vocab 16K, 32/64 experts, split words, reversed rows).

**Where the table loses** (width 8, 2 layers, vocab 500, 2,400 steps, 124 tokens per weight):

| Reversed rows | Learned | Hash both | Hash L0 | Hash L1 |
|---|---|---|---|---|
| 0% | 2.911 | 2.910 | | |
| 20% | 2.976 | 2.985 | | |
| 50% | 3.129 | 3.154 | 3.131 | 3.124 |
| 50%, shared top-2 | 3.117 | 3.144 | | |

Gap along training at 50%: -0.032, -0.009, -0.006, 0.000, +0.023, +0.025. Test-time context cost rose to +0.009 (L1 +0.007). One learned layer wins it all back. 4 layers at the same steps: all arms tie (learned 3.122, first 3 hashed 3.119, last 3 hashed 3.128, all hashed 3.120).

Takeaway: hash the first layer(s) for the early gain; keep learned routers later, because the table only loses when words need context-dependent experts after long training, and one learned layer fixes that. Paper claim 7.

## How to fill the table (seventh round)

Question: which word-to-expert assignment trains best? 2 layers, width 32, vocabulary 4,000, 200 steps, top-1, no shared expert, all layers hashed. New flags: `--hashtype sorted|cluster|clusterbal`, `--pairhash N`.

| Table | 16 experts | 32 experts | 64 experts | 64 experts, vocab 16,000 |
|---|---|---|---|---|
| Random | 5.071 | 5.070 | 5.066 | 6.629 |
| Sorted by frequency, cut into equal-frequency runs | 5.058 | 5.044 | 5.046 | 6.619 |
| Grouped by next-word use (k-means) | 5.066 | | 5.044 | |
| Grouped by use, then equal-frequency cut | 5.070 | 5.051 | 5.038 | 6.637 |

- The sorted table beats random at every point and has the lowest load (busiest/average 3.11 vs 3.46 at 64 experts). The gain sits after rare input words (rank 1,001+: 5.185 vs 5.258).
- Greedy balanced mixing equals random. Grouping by use is inconsistent and, without the cut, overloads one expert 7.9 times the average.
- With a shared expert and top-2, the method does not matter (5.045 vs 5.047). In long training (2,400 steps) sorted still wins (3.399 vs 3.417).
- Spreading common words over experts by the previous token costs loss on web text (all words: 5.184 vs 5.065). On half-reversed text it recovers part of the table's loss (3.139 vs 3.154; learned 3.129).

Takeaway: keep each word on one expert set, sort words by frequency and cut into runs of equal total frequency. Paper claim 6.

## Repeats and two fixes (eighth round, 2026-11-13)

The key short comparisons were run 3 times each, then two fixes were tested: send the most common words only to the shared expert (`--sharedonly N`), and start with a table and switch to a learned router later (`--switch S`).

| Test | Result |
|---|---|
| Table vs learned, 16 / 32 experts, top-1 (3 runs each) | -0.094 / -0.103, spread between repeats at most 0.039 |
| Table vs learned, shared expert, top-2 of 32 (3 runs each) | -0.036, spread at most 0.012 (at 8 experts it ties) |
| Sorted vs random table, 16 / 32 experts (3 runs each) | sorted wins all 6 pairs, by 0.013 to 0.031 |
| Top 3 / top 10 words to the shared expert only, E/k = 64 | balanced up to E/k 42 / 87 instead of 26; loss +0.006 / +0.018 |
| Table then learned router, switch at 600 or 1,200 of 2,400 steps | 3.167 to 3.211, worse than table (3.154) and learned (3.129) |

- The table's early lead and the sorted table's gain hold over repeats.
- With a shared expert and top-2, the table still leads at 32 experts; the tie needs few experts as well.
- Sending "the", "," and "." to the shared expert lifts the load limit at no measurable loss. Removing only "the" does nothing, because "," and "." are almost as common.
- Switching from a table to a learned router does not combine their strengths, even when the router is trained to copy the table first.

Takeaway: the conclusions survive repeats, and the load limit can be lifted by letting the shared expert alone handle the few most common words.

## Moving toward large models (ninth round, 2026-11-13, `scale.py`)

One condition at a time moved toward large models, from 16 experts, top-1, no shared expert, width 32, 2 layers, vocabulary 4,000, 200 steps.

| Condition moved | Table minus learned | Sorted minus random |
|---|---|---|
| None (3 runs each) | -0.094 | -0.020 |
| Vocabulary 16,000 | -0.095 | -0.038 |
| 4 layers / context 192 / head size 32 / learning rates 2x | -0.115 / -0.085 / -0.082 / -0.107 | -0.018 / -0.013 / -0.023 / -0.016 |
| Batch 64, 50 steps / 64 experts | -0.146 / -0.124 | -0.024 / -0.017 |
| 600 steps / shared expert top-2 of 32 | -0.053 / -0.030 | -0.015 / -0.006 |
| Width 64 / width 128 | -0.034 / +0.007 | -0.028 / - |
| Vocabulary 16,000 + width 64 + 4 layers + shared top-2 of 32 | -0.043 | +0.015 |

- Every condition except width kept the order: learned router worst, sorted table best.
- Width removes the table's lead: a wide model's router moves words just as often (churn 0.45 at width 128 vs 0.40 at width 32), but forcing 50% of words to move costs +0.039 at width 128 against +0.233 at width 32.
- Sending the top 3 words to the shared expert still costs nothing measurable at vocabulary 16,000 (6.593 vs 6.600).

Takeaway: in large, wide models the fixed table's early loss advantage should be small; its value is the load rule and needing no router.
