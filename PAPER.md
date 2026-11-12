# Hash Routing: When Should Each Token in the Vocabulary Have Its Own Fixed Expert?

Vuk Rosić

## Abstract

Give every word of the vocabulary its own fixed expert, and the experts that hold "the" do all of its work: about one token in 26 of English web text, because a table cannot split a word between experts. So the busiest expert carries at least the most common word's share times the number of experts per chosen expert. With 64 experts per chosen one, as in DeepSeek-V4's 384 experts with 6 chosen, that is 2.4 times the average. Below 26 experts per chosen expert, a table balanced by word frequency spreads the work evenly; above it, no table can. Only this ratio matters, not the number of experts: 64 experts with 1 chosen and 384 with 6 chosen give the same busiest expert. Under a capacity limit, the work that overflows is the sum, over the common words, of how far each word's share exceeds its experts' capacity. So the whole load cost of a fixed table is known from word counts before training.

<img src="figures/abs_load.svg" width="100%">

*Figure 1. A table cannot split a word, so the experts of the most common words are the busiest. Past 26 experts per chosen expert no table can balance, and only that ratio matters.*

The same limit binds a learned router whenever it routes by word, and in our models it does: a learned router sent two thirds or more of each word's uses to one expert, and its busiest expert sat at the same limit. In loss, the fixed table never trains worse than a learned router. It is ahead early in training, because a learned router keeps moving words between experts. Each move takes a word away from the expert that has learned it and hands it to one that has not. The moves do not come from the router learning: a router frozen at its random start moves words just as often, because its input, the hidden state, keeps changing as the rest of the model trains. The lead shrinks as training goes on and is gone once each weight has seen a few hundred tokens with few experts; with many small experts it lasts longer.

<img src="figures/abs_table.svg" width="100%">

*Figure 2. A learned router keeps moving a word between experts during training. A fixed table never moves it.*

<img src="figures/abs_cause.svg" width="100%">

*Figure 3. The early loss follows the moves. Making a fixed table move words on purpose costs loss in proportion to how many it moves.*

The early lead is largest with one expert per word, no shared expert and many experts, and it vanishes with a shared expert and several experts per word, because a move only damages the share of the word's output that the moved expert carries. A random table and a frequency-balanced table give the same loss, so a table should be balanced by word frequency, which only lowers the load.

<img src="figures/abs_rules.svg" width="100%">

*Figure 4. When the fixed table helps early in training, and how to build it. Bar heights show the shape of each trend.*

We prove the load limit, compute it exactly from the word counts of two tokenizers in 14 expert designs each, and check it in training, where the predicted overflow and busiest load match the measured ones. We show that the moves cause the early loss by forcing a fixed table to move words and by freezing the router's weights. We imitate long training with tiny models that see up to 248 tokens per weight, and repeat the key runs to measure noise.

## 1. What hash routing is

A mixture-of-experts (MoE) feed-forward layer replaces one feed-forward block with E smaller blocks called experts, plus, in many designs, one shared expert that every token uses. A router picks k of the E experts for each token:

```text
scores = sqrt(softplus(W_r h))          # one score per expert, from the hidden state h
chosen = top_k(scores + bias)           # learned routing
y      = shared(h) + sum_{e in chosen} w_e * expert_e(h)
```

Hash routing replaces the `top_k` line with a table fixed before training:

```text
chosen = table[token_id]                # k experts per vocabulary token, fixed once
```

The router can still set the weights `w_e` of the chosen experts, as in DeepSeek-V4. In "The cat sat", the position of " cat" always uses the experts in row " cat" of the table, in every sentence and at every training step. A learned router picks from the hidden state, so the same word can go to different experts in different sentences and, as Section 9 shows, at different steps of training.

<img src="figures/moe_hash.svg" alt="Learned routing and hash routing" width="100%">

**Figure 5.** (a) Learned routing: the router reads the hidden state (1), picks top-k experts (2), and the output is the shared expert plus the chosen experts (3). (b) Hash routing: the token ID picks the experts from a fixed table (1); the router only weights them (2).

The table costs no compute and `V x k` small integers per layer. Two ways to fill it:

```text
random:    each token gets k experts drawn at random
balanced:  tokens from most to least common, each gets the k experts with the least total frequency so far
```

## 2. Who uses it today

| User | How | What they report |
|---|---|---|
| Hash Layers, Roller et al. 2021 | Top-1 hash in one layer, no shared expert | Beats Switch routing; the lead grows from 16 to 128 experts; random table about equal to a balanced one. Suggests in one sentence that learned routing is hard because each expert's members change while it learns |
| StableMoE, Dai et al. 2022 | Learn a router for 10% of training, then freeze a distilled copy | Names "routing fluctuation": 41% of tokens still change expert after 20% of training. The frozen router beats Hash Layers (18.93 vs 19.25 PPL, 124M backbone, top-1) |
| Clark et al. 2022 | Hash vs learned routers up to hundreds of billions of weights | Hash worse than a balanced learned router (S-BASE) at most sizes |
| OpenMoE, Xue et al. 2024 | Analysis of a trained MoE | Learned routing is mostly by token ID and settles early |
| Fan, Messmer, Jaggi 2024 | Frozen random router | About equal to a learned router |
| DeepSeek-V4, 2026 | Hash routing in the first 3 MoE layers; 1 shared expert plus 6 of 256 or 384 routed experts | No reason given |

These papers report that hash routing is competitive in loss. None gives the load a fixed table must carry, which design choices set it, or when a table can be balanced at all. None shows why a table can lead early, and none tests whether expert moves cause the loss; StableMoE's evidence is correlational.

## 3. The claims

| # | Claim | Section |
|---|---|---|
| 1 | A fixed table's busiest expert carries at least max(1, p E/k) times the average, where p is the most common token's share. A frequency-balanced table reaches this limit, and only the ratio E/k matters | 5 |
| 2 | Under a capacity limit, a balanced table overflows by the sum over tokens of max(0, p_i - c k/E); a random table overflows more | 6 |
| 3 | A learned router that routes by word hits the same limit | 7 |
| 4 | In loss, a fixed table never trains worse than a learned router; it leads early and ties after long training | 8 |
| 5 | The early lead comes from the learned router moving words; a move costs more the more of the word's output one expert carries, and the more experts there are | 9 |
| 6 | Random and balanced tables give the same loss, so balance the table for load | 10 |

Claims 1 and 2 are proven for any token frequencies and checked exactly on two tokenizers and in training. Claims 3, 4 and 6 held in every setting tested. Claim 5's sizes are trends over 8 designs.

## 4. Setup

**Load, without training (`rule.py`).** Token frequencies of FineWeb web text under two tokenizers: GPT-2 (50K tokens, counts over the whole training set) and cl100k (100K tokens, 3M tokens of the same text re-tokenized). The most common token is 3.82% and 3.78% of text, so 1/p is about 26 for both. For 14 designs from 8 experts with 1 chosen to 768 with 6 chosen, we build a random and a balanced table and compute each expert's exact share of the routed work, for an infinitely large batch.

**Model.** Pre-norm transformer with RMSNorm, RoPE, QK-norm, no biases, untied output head. Every feed-forward block is a DeepSeek-style MoE: sqrt(softplus) router scores, aux-loss-free balancing (a per-expert bias moved by 0.01 per step toward equal load), router weights renormalized over the chosen experts. Active feed-forward width equals a dense SwiGLU block; with a shared expert, the shared expert takes half of it and the k routed experts share the rest. Muon (Newton-Schulz, momentum 0.95, lr 0.02) trains block matrices, one expert matrix at a time; Adam (betas 0.9, 0.95, lr 0.01) trains embeddings, head, norms and router.

**Data.** FineWeb (GPT-2 tokens) mapped to the V-1 most common tokens plus an unknown token. In the short runs, planted ParaRel fact sentences fill 10% of rows.

**Short runs.** Width 32, 2 layers, 4 heads, vocabulary 4,000, 16 rows of 48 tokens per step, 200 steps, warmup 20 steps, linear cooldown over the last 30%. 3 to 7 s on a laptop CPU.

**Long runs.** To imitate the many tokens each weight sees in a large model, we shrink the model: width 8 or 16, vocabulary 500 or 4,000, batches of 16 to 128 rows, up to 2,400 steps. These models see 37 to 248 tokens per weight in 15 to 30 s (DeepSeek-V3 saw about 22 tokens per total weight). A small vocabulary maps rare words to one unknown token, 42% of text at vocabulary 500 and 20% at 4,000, which no real model has. A table would send it to one expert. So in long loss runs the unknown token uses the learned router in hashed layers, and in load and capacity runs it uses only the shared expert and is left out of load in both arms.

**Scores.** Web loss on 300 held-out FineWeb rows. Churn: the share of held-out positions whose set of experts changed between two checkpoints, averaged over layers. Token share: the share of a word's uses (words seen 5 or more times) that go to its most common expert set. Busiest / average: the busiest expert's load divided by the mean, per training batch, averaged over the last quarter of training. Dropped: the share of routed work over capacity.

**Tests.**

| Test | What it tells |
|---|---|
| Exact load of random and balanced tables, 2 tokenizers, 14 designs | Claims 1, 2 |
| Measured load and overflow in training vs the prediction from word counts | Claims 1, 2 |
| Capacity limits 1.0, 1.25, 2.0 with random, balanced and learned routing | Claim 2's cost in loss |
| Busiest expert of learned routers in every layer of 2- and 4-layer models | Claim 3 |
| Table vs learned in 8 designs, 13 other settings, and long runs to 248 tokens per weight | Claim 4 |
| Churn of a normal, slowed, frozen and word-only router; forced moves of a table in 8 designs | Claim 5 |
| Random, balanced and learned-then-frozen tables | Claim 6 |

**Noise.** Three repeated runs of the same setting differed by up to 0.02 in web loss in the short runs and by up to 0.015 in the long 64-expert runs, so smaller single-run differences are not ordered.

## 5. Claim 1: a table's busiest expert carries at least p E/k times the average

**Why.** A table sends every use of a token to the same k experts. The most common token, with share p of all text, puts p/k of all routed work on each of its experts. A fair share is 1/E. So the busiest expert carries at least (p/k) / (1/E) = p E/k times the average, whatever the table. When p E/k is below 1, the limit is 1 and a table can be balanced in principle. The limit depends on E and k only through E/k.

**Test.** Exact load from the word counts of two tokenizers, for random and balanced tables.

<img src="figures/load_rule.svg" width="100%">

**Figure 6.** Left: busiest expert / average for 14 designs per tokenizer. Lines are the limit max(1, p E/k); filled dots are balanced tables, open dots random tables. Right: share of routed work over a capacity of 1.25 times fair; the line is the sum formula of claim 2.

| Experts, chosen | E/k | Limit | Balanced table | Random table |
|---|---|---|---|---|
| 8, 1 | 8 | 1.00 | 1.00 | 1.20 |
| 16, 1 | 16 | 1.00 | 1.00 | 1.73 |
| 64, 4 | 16 | 1.00 | 1.00 | 2.30 |
| 32, 1 | 32 | 1.22 | 1.27 | 3.05 |
| 256, 8 | 32 | 1.22 | 1.27 | 3.12 |
| 64, 1 | 64 | 2.44 | 2.49 | 5.65 |
| 128, 2 | 64 | 2.44 | 2.49 | 4.57 |
| 256, 4 | 64 | 2.44 | 2.49 | 3.83 |
| 384, 6 | 64 | 2.44 | 2.49 | 3.62 |
| 128, 1 | 128 | 4.88 | 4.93 | 5.74 |
| 768, 6 | 128 | 4.88 | 4.93 | 6.33 |
| 256, 1 | 256 | 9.77 | 9.82 | 10.76 |

(GPT-2 tokens. cl100k gives limits about 1% lower and the same pattern; every row of both is in `rule.json`.)

The balanced table sits 0.05 above the limit in every design, under both tokenizers. Designs with the same E/k give the same balanced load to two decimals, from 64 experts with 1 chosen to 384 with 6. A random table is always worse and varies with the draw: at E/k = 64 it ranged from 3.1 to 6.3 over 8 tables.

**In training.** 64 or 16 experts, 1 chosen, shared expert, vocabulary 4,000 (top real token 4.8%), 400 steps of 64 rows. The prediction uses the exact table of each layer and no training.

| Table | Layer | Busiest / average, predicted | Measured |
|---|---|---|---|
| Random, 64 experts | 0 / 1 | 3.63 / 3.99 | 3.86 / 4.09 |
| Random, 16 experts | 0 / 1 | 2.33 / 1.55 | 2.34 / 1.64 |
| Balanced, 64 experts | 0 / 1 | 3.07 (limit) | 3.27 / 3.27 |

Measured loads are 0.01 to 0.23 above the prediction; the extra comes from batch-to-batch noise with about 48 tokens per expert per batch.

## 6. Claim 2: the overflow under a capacity limit is set by word counts

**Why.** With capacity c times the fair share, each expert takes at most c/E of the routed work. A balanced table gives each very common token experts of its own, so token i overflows by k (p_i/k - c/E) = p_i - c k/E whenever that is positive. The total overflow is the sum of max(0, p_i - c k/E) over tokens. It again depends only on E/k.

**Test.** The formula against the exact overflow (Figure 6, right) and against training.

| E/k | Formula, capacity 1.25 | Balanced table, exact | Random tables, exact |
|---|---|---|---|
| 16 | 0 | 0 | 3.0 to 7.7% |
| 32 | 0 | 0.1% | 7.7 to 11.4% |
| 64 | 5.4% | 5.8% | 14.1 to 18.3% |
| 128 | 12.9% | 13.3% | 19.7 to 21.8% |
| 256 | 19.4% | 19.8% | 25.4 to 26.9% |

The formula is within 0.4 points of the exact balanced overflow. In training (same setup as Section 5), the predicted overflow of the random table was 20.1% at 64 experts and 9.7% at 16; the measured one was 20.6% and 9.8%.

**Cost in loss.** Overflowing work is dropped: its expert output is zeroed and the residual stream passes the token on.

| 64 experts, 1 chosen, shared | Capacity | Dropped | Web loss | Cost |
|---|---|---|---|---|
| Random table | none / 2.0 / 1.25 / 1.0 | 0 / 9.8 / 20.6 / 26.8% | 4.835 / 4.853 / 4.856 / 4.855 | 0.018 to 0.021 |
| Balanced table | none / 1.25 | 0 / 10.5% | 4.833 / 4.835 | 0.002 |
| Learned router | none / 1.25 | 0 / 15.2% | 4.875 / 4.894 | 0.019 |

Dropping cost at most 0.021 loss in these runs, and a balanced table's drops cost nothing measurable: they fall on its few very common tokens, while a random table's busy experts also drop many ordinary words. In a short 200-step run with vocabulary 16,000, the random table's drops cost 0.044 and the balanced table's 0.027. Without a capacity limit nothing is dropped and the same imbalance costs step time instead: the expert part of a step takes as long as the busiest expert.

## 7. Claim 3: a learned router that routes by word hits the same limit

**Why.** The limit of claim 1 holds for any router whose choice is set by the token. A learned router can go below it only by sending the same word to different experts in different contexts.

**Test.** Busiest / average of learned routers in the last quarter of training, 64 experts, 1 chosen, shared expert, vocabulary 4,000 (limit 3.07).

| Model | Layer | Token share | Busiest / average |
|---|---|---|---|
| 2 layers, 400 steps | 0 / 1 | 0.94 / 0.65 | 3.68 / 3.15 |
| 4 layers, 250 steps | 0 / 1 / 2 / 3 | 0.80 / 0.80 / 0.74 / 0.74 | 3.13 / 3.27 / 3.59 / 3.18 |

Every layer sends two thirds or more of a word's uses to one expert and sits at the limit, as high as the tables. Faster bias balancing (5 times the step) did not lower the busiest expert in a 192-expert design. So in these models the fixed table costs no load compared with the learned router. OpenMoE reports that large learned routers also route mostly by token, which makes the limit relevant beyond our size, but a deep layer of a large model may split common words by context; we do not measure that.

## 8. Claim 4: in loss, a fixed table never trains worse than a learned router

**Why.** A learned router has to find a good assignment while the experts learn. A fixed table gives every expert the same words from step one.

**Test 1: 8 designs, short runs.** All layers learned vs all layers hashed.

| Design | Learned | Fixed table | Table minus learned |
|---|---|---|---|
| No shared expert, top-1, 4 experts | 5.116 | 5.086 | -0.030 |
| No shared expert, top-1, 8 experts | 5.144 | 5.065 | -0.079 |
| No shared expert, top-1, 16 experts | 5.185 | 5.081 | -0.104 |
| No shared expert, top-2 of 8 | 5.095 | 5.073 | -0.023 |
| No shared expert, top-4 of 8 | 5.086 | 5.075 | -0.012 |
| Shared expert, top-1 of 8 | 5.106 | 5.059 | -0.047 |
| Shared expert, top-2 of 8 | 5.070 | 5.068 | -0.003 |
| Shared expert, top-4 of 8 | 5.080 | 5.073 | -0.007 |

The same held when one condition at a time moved toward larger models (base: no shared expert, top-1 of 8):

| Setting | Table minus learned |
|---|---|
| Base | -0.079 |
| Vocabulary 1,000 / 500 | -0.068 / -0.060 |
| Context 16 / 96 (tokens per step fixed) | -0.089 / -0.092 |
| Learning rate 2x / 1/4, both models | -0.093 / -0.082 |
| Cooldown over 80% of training | -0.088 |
| Router learning rate 0.1x / 0 / 3x | -0.074 / -0.081 / -0.077 |
| Half the training rows reversed (same words, different job), shared expert top-2, 400 steps | -0.008 |

**Test 2: long training.** Tiny models trained until each weight has seen as many tokens as in a large model.

| Model | Design | Tokens per weight | Table minus learned at end |
|---|---|---|---|
| Width 16, vocabulary 500 | 8 experts, top-1, no shared | 37 | -0.006 |
| Width 8, vocabulary 500 | 8 experts, top-1, no shared | 124 | -0.001 |
| Width 8, vocabulary 500 | 8 experts, top-1, no shared | 248 | -0.001; repeat -0.002 |
| Width 16, vocabulary 500 | 64 experts, top-4, shared | 35 | -0.001 |
| Width 8, 4 layers, vocabulary 500 | 8 experts, top-1; hash layer 0 or 3 only | 124 | -0.004 / -0.001 |
| Width 8, vocabulary 500 | 192 experts, top-3, shared (E/k = 64) | 81 | -0.017 |
| Width 8, vocabulary 4,000 | 64 experts, top-1, shared | 14 | -0.037 (mean of 3 runs each; spread 0.012 / 0.015) |
| Width 8, vocabulary 4,000 | 16 experts, top-1, shared | 17 | -0.029 |

In the 248-tokens-per-weight run the gap along training was -0.064, -0.022, -0.013, -0.005, -0.002 and -0.001 at steps 100 to 600: it shrinks to zero and does not cross. With many small experts the lead lasts longer (-0.017 at 81 tokens per weight with 192 experts). Tokens seen set the gap, not steps: 300 steps of 128 rows gave the same gap as 2,400 steps of 16. The learned router never finished ahead by more than noise.

**Where it is weaker.** Clark et al. found a balanced learned router ahead of hash routing at most sizes up to hundreds of billions of weights. Our longest runs tie, so whether the learned router pulls ahead at real scale is not settled by tiny models.

## 9. Claim 5: why the table leads early

### 9.1 A learned router keeps moving words, because its input changes

**Why.** The router scores experts from the hidden state. The hidden state of the same word changes as attention and earlier layers train, so the top-scoring expert can change even if the router's weights do not.

**Test.** Churn between checkpoints 100 steps apart, with the router's weights trained normally, slowed, frozen at the random start, or reading only the word embedding.

| Router (no shared expert, top-1 of 8) | Churn | Token share | Web loss |
|---|---|---|---|
| Normal | 0.54 | 0.70 | 5.144 |
| Router learning rate 0.1x | 0.52 | 0.71 | 5.139 |
| Frozen at random start | 0.53 | 0.69 | 5.146 |
| Router learning rate 3x | 0.54 | 0.64 | 5.142 |
| Reads only the word embedding | 0.32 | 1.00 | 5.127 |
| Fixed table | 0 | 1.00 | 5.065 |

A frozen router moved words as often as a learning one and lost as much. A router that sees only the word still moves words, because the embedding keeps training, and is still 0.062 behind the table. Churn stays high in long runs (0.3 to 0.5 at 124 tokens per weight; 0.84 to 0.99 with 192 experts), but its cost falls (Section 9.2).

### 9.2 The moves cause the early loss

**Test.** Every 25 steps, re-assign a random share of the table's words to new random experts.

| No shared expert, top-1 of 8 | Churn | Web loss |
|---|---|---|
| Fixed table | 0 | 5.065 |
| Table, 10% re-assigned every 25 steps | 0.20 | 5.110 |
| Table, 50% re-assigned every 25 steps | 0.81 | 5.262 |
| Learned router | 0.54 | 5.144 |

Forced moves cost loss in proportion to the churn they cause, and the learned router sits between the two forced tables. Learned moves are cheaper than random ones: across 8 designs, forced-move cost times learned churn predicts the learned router's gap with correlation 0.92 but about 1.6 times too large. Late in training they become nearly free: at 124 tokens per weight the learned router still moves 30 to 50% of positions and ties, while a table forced to move half its words still loses 0.05 to 0.06.

### 9.3 A move costs more the more one expert carries

**Test.** Forced moves (50% every 25 steps) in each design. Weight per expert is the share of the feed-forward width one routed expert holds: 1/k without a shared expert, 1/(2k) with one.

| Design | Weight per expert | Forced churn | Loss per unit of churn |
|---|---|---|---|
| No shared expert, top-1 | 1 | 0.81 | 0.243 |
| No shared expert, top-2 | 1/2 | 0.87 | 0.103 |
| Shared expert, top-1 | 1/2 | 0.81 | 0.067 |
| No shared expert, top-4 | 1/4 | 0.92 | 0.044 |
| Shared expert, top-2 | 1/4 | 0.87 | 0.033 |
| Shared expert, top-4 | 1/8 | 0.92 | 0.024 |

Loss per unit of churn follows the weight per expert (correlation 0.88 over 8 designs). With a shared expert and top-2, the same churn costs 7 times less than with no shared expert and top-1, which is why the learned router ties there. At equal weight per expert, the shared-expert design is cheaper than the top-2 design (0.067 vs 0.103), so a shared expert protects more than its width alone predicts.

### 9.4 A move costs more with more experts

| Experts (no shared, top-1) | Forced churn | Loss per unit of churn | Learned churn | Table minus learned |
|---|---|---|---|---|
| 4 | 0.70 | 0.116 | 0.40 | -0.030 |
| 8 | 0.81 | 0.243 | 0.54 | -0.079 |
| 16 | 0.88 | 0.264 | 0.65 | -0.104 |

With more experts each expert sees a narrower set of words, so a moved word lands where nothing like it has been learned, and the learned router also moves more. Hash Layers saw the same growth from 16 to 128 experts.

## 10. Claim 6: balance the table by word frequency

**Why.** If stability is what matters for loss, any fixed table should do; balancing changes only the load.

**Test.**

| Table | Web loss, 8 experts top-1 (short) | Web loss, 64 experts top-1 shared (long) | Busiest / average at 64 experts | Dropped at capacity 1.25 |
|---|---|---|---|---|
| Random | 5.065 | 4.835 | 3.86 / 4.09 | 20.6% |
| Balanced by frequency | 5.077 | 4.833 | 3.27 / 3.27 | 10.5% |
| Learned 50 steps, then frozen | 5.091 | | | |
| Learned 100 steps, then frozen | 5.101 | | | |
| Learned throughout | 5.144 | 4.875 | 3.68 / 3.15 | 15.2% |

Random and balanced tables give the same loss within noise. Balancing halves the overflow and brings the busiest expert to the limit of claim 1. Freezing a learned router's choices helps, but is no better than a table fixed from step one. Hash Layers also found random and balanced tables equal in loss.

**Where it is weaker.** StableMoE's frozen learned router beat Hash Layers at a 124M backbone. A learned assignment may gain value with model size and training length.

## 11. Who should use it

| Case | Recommendation | Claim |
|---|---|---|
| E/k at most 1/p (about 26 for English web text) | A balanced fixed table spreads work evenly and loses no loss | 1, 4, 6 |
| E/k above 1/p, with a capacity limit | The table overflows by the sum of max(0, p_i - c k/E): 5.4% at E/k = 64 and capacity 1.25. Balance it, or raise the capacity of the experts that hold the most common tokens | 2, 6 |
| E/k above 1/p, dropless | The expert part of each step takes p E/k times the balanced time (2.4x at DeepSeek-V4's 384 / 6) | 1 |
| Early layers, where a learned router routes by word anyway | A table costs no extra load and no loss | 3, 4 |
| One expert per word, no shared expert, many experts | A table also trains faster early | 5 |
| Shared expert and several experts per word, long training | Either; they tie in loss | 4, 5 |
| Building the table | Balance by frequency; random gives the same loss but more overload | 6 |

## 12. Limits

- All trained models are under 2M weights; the long runs are under 0.1M weights and reach 248 tokens per weight. Large-model behaviour is imitated with tiny models, never run.
- Claims 1 and 2 hold for any token frequencies, but p depends on the tokenizer and data. DeepSeek-V4 uses its own 128K tokenizer on mixed data, so its exact numbers differ.
- Claim 3 is measured in models whose learned routers route mostly by word in every layer. Deep layers of large models may split common words by context and balance below the limit.
- The loss cost of overflow (claim 2) is measured only in tiny models, where it was at most 0.021 at 27% dropped.
- Most comparisons are single runs; the key long comparisons were repeated two or three times.

## Appendix A. Every short-run setting

Base unless noted: width 32, 2 layers, 200 steps, batch 16, 8 experts. Web loss.

| Setting | Learned | Fixed table | Table minus learned | Learned churn |
|---|---|---|---|---|
| No shared, top-1, 8 experts (base) | 5.144 | 5.065 | -0.079 | 0.54 |
| No shared, top-1, 4 experts | 5.116 | 5.086 | -0.030 | 0.40 |
| No shared, top-1, 16 experts | 5.185 | 5.081 | -0.104 | 0.65 |
| No shared, top-2 | 5.095 | 5.073 | -0.023 | 0.74 |
| No shared, top-4 | 5.086 | 5.075 | -0.012 | 0.77 |
| Shared, top-1 | 5.106 | 5.059 | -0.047 | 0.49 |
| Shared, top-2 | 5.070 | 5.068 | -0.003 | 0.75 |
| Shared, top-4 | 5.080 | 5.073 | -0.007 | 0.72 |
| Base, vocabulary 1,000 | 3.639 | 3.572 | -0.068 | 0.56 |
| Base, vocabulary 500 | 3.061 | 3.001 | -0.060 | 0.61 |
| Base, context 16, batch 48 | 5.170 | 5.081 | -0.089 | 0.59 |
| Base, context 96, batch 8 | 5.193 | 5.101 | -0.092 | 0.56 |
| Base, learning rate 2x | 5.139 | 5.046 | -0.093 | 0.62 |
| Base, learning rate 1/4 | 5.340 | 5.258 | -0.082 | 0.44 |
| Base, cooldown 80% | 5.192 | 5.104 | -0.088 | 0.43 |

## Appendix B. Load of every design

Exact busiest / average and overflow at capacity 1.25 for both tokenizers are in `rule.json`, produced by `python3 rule.py` in 10 s.

## References

- Roller, Sukhbaatar, Szlam, Weston. Hash layers for large sparse models. NeurIPS 2021, arXiv 2106.04426.
- Dai et al. StableMoE: stable routing strategy for mixture of experts. ACL 2022, arXiv 2204.08396.
- Clark et al. Unified scaling laws for routed language models. ICML 2022, arXiv 2202.01169.
- Xue et al. OpenMoE: an early effort on open mixture-of-experts language models. ICML 2024, arXiv 2402.01739.
- Fan, Messmer, Jaggi. Towards an empirical understanding of MoE design choices. arXiv 2402.13089.
- Ternovtsii, Bilak. Equifinality in mixture of experts. arXiv 2604.14419.
- Dikkala et al. On the benefits of learning to route in mixture-of-experts models. EMNLP 2023.
- DeepSeek-AI. DeepSeek-V3 technical report. arXiv 2412.19437.
- DeepSeek-AI. DeepSeek-V4 technical report. arXiv 2606.19348.
- Fedus, Zoph, Shazeer. Switch Transformers. arXiv 2101.03961.
- Penedo et al. FineWeb. arXiv 2406.17557. Elazar et al. ParaRel. arXiv 2102.01017.
- Jordan et al. Muon, 2024; Liu et al. Muon is scalable. arXiv 2502.16982.

## Code

`hr.py` (test bed, every flag in `--help`), `rule.py` (exact load, claims 1 and 2), `load.py`, `curve.py` (gap along training), `sweep.py`, `table.py`, `fit.py`, and `results.jsonl` with every run.
