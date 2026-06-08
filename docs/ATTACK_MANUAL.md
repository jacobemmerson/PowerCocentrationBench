# Attack Manual

This is the canonical reference spec for every attack in PowerConcentrationBench. It exists so
that "what does this attack actually do, and how do we know an implementation is correct" has
one authoritative answer — independent of any single implementer's interpretation.

Each entry documents, in implementation-ready terms:
- **Algorithm** — the core idea and step-by-step procedure.
- **Access level** — which `attacks/` category it belongs to, and exactly what it needs from
  `models/loader.py` (chat-only, logits, gradients, hidden states, weights + training loop).
- **Key hyperparameters** — the knobs that should map directly onto the `configs/` schema.
- **Success signal** — what the attack's *own* search/optimization loop uses to know it's
  converging. This is deliberately distinct from the benchmark's reported result, which always
  comes from the configured judges (StrongREJECT / HarmBench / LLM-as-judge / multi-judge).
- **Citation** — the source paper(s) an implementation should match.
- **Pitfalls** — gotchas that would otherwise be rediscovered the hard way.

This manual is owned collectively: whichever agent implements an attack writes and maintains
its entry, verified against the cited paper(s), so every other agent (and future contributor)
can treat it as ground truth rather than re-deriving the algorithm from scratch. If an
implementation reveals the manual is wrong or incomplete, fixing the manual is part of the change.

---

## Prompt-based / black-box attacks
*(owned by the prompt-attack-agent — `attacks/prompt_based/`)*

### AutoDAN (Genetic Algorithm)
- **Algorithm**: Initialize a population from handcrafted DAN-style jailbreak templates. Each
  generation: score every member's fitness (negative log-likelihood of an affirmative target
  response if logits are available — a grey-box variant — or a judge score if not), select
  parents (roulette/tournament selection), apply crossover (recombine sentence-level segments
  from two parents) and mutation (use an LLM to paraphrase/rewrite segments while preserving
  intent), and replace the lowest-fitness members. Repeat until a member jailbreaks the target
  or the generation budget is exhausted.
- **Access level**: Black-box (chat only); can optionally use target logits for fitness if the
  target is white-box, but does not require gradients or weights.
- **Key hyperparameters**: population size, number of generations, crossover rate, mutation
  rate, elite-retention count, fitness-function choice (loss-based vs. judge-based).
- **Success signal**: fitness score (log-likelihood of an affirmative continuation, or judge
  score) crossing a configured threshold.
- **Citation**: Liu, X., et al. (2024). *AutoDAN: Generating Stealthy Jailbreak Prompts on
  Aligned Large Language Models.* ICLR 2024.
- **Pitfalls**: LLM-based mutation is expensive — cache and batch calls. Loss-based fitness
  needs white-box logits (grey-box setting); judge-based fitness is purely black-box but slower
  and costlier per evaluation. Populations can collapse onto repetitive templates — track
  diversity explicitly rather than trusting "best fitness improved" alone.

### AutoDAN (Hierarchical Genetic Algorithm)
- **Algorithm**: Two-level extension of AutoDAN-GA. An outer GA evolves whole "paragraph"-level
  jailbreak scaffolds; an inner GA evolves "sentence"-level building blocks within each
  scaffold; periodic migration moves high-fitness building blocks between levels. The
  hierarchical structure increases template diversity and convergence speed versus a flat GA.
- **Access level**: Black-box (same requirements as AutoDAN-GA).
- **Key hyperparameters**: outer/inner population sizes, migration interval and rate, per-level
  mutation/crossover rates, generations per level.
- **Success signal**: same fitness-threshold convergence as AutoDAN-GA, tracked at both levels.
- **Citation**: Liu, X., et al. (2024). *AutoDAN: Generating Stealthy Jailbreak Prompts on
  Aligned Large Language Models.* ICLR 2024 (hierarchical variant).
- **Pitfalls**: substantially more LLM calls than the flat GA — budget carefully. Inner loops
  can overfit to a single outer scaffold. Nested-population bookkeeping (which inner population
  belongs to which outer member, after migration) is easy to get subtly wrong — write tests for
  the migration step specifically.

### PAIR (Prompt Automatic Iterative Refinement)
- **Algorithm**: An attacker LLM, given the goal and the target's previous response, proposes a
  refined candidate jailbreak prompt. The candidate is sent to the target; an LLM judge scores
  the response (typically 1–10); the score and response feed back to the attacker for the next
  round. Multiple independent "streams" (different seeds/initial framings) run in parallel to
  improve coverage and reduce reliance on any single refinement trajectory.
- **Access level**: Black-box (attacker LLM + target chat + judge; no gradients or weights).
- **Key hyperparameters**: number of parallel streams, max iterations per stream, attacker
  sampling temperature, judge-score early-stopping threshold.
- **Success signal**: judge score reaching the configured threshold (e.g. 10/10) on any stream.
- **Citation**: Chao, P., et al. (2023). *Jailbreaking Black Box Large Language Models in
  Twenty Queries.* arXiv:2310.08419.
- **Pitfalls**: the attacker LLM may itself refuse to produce jailbreak content — detect this
  explicitly rather than scoring it as a failed attack against the target. Three-way
  conversation-state management (attacker / target / judge) is easy to tangle — keep each
  role's history separate and explicit. Reported query-efficiency gains depend on early stopping
  actually firing — verify it does.

### TAP (Tree of Attacks with Pruning)
- **Algorithm**: Extends PAIR into a tree search. Each node is a candidate prompt; branching
  generates several PAIR-style refinements per node; an evaluator LLM scores each branch on both
  "on-topic-ness" and harmfulness, pruning low-scoring or off-topic branches *before* they
  consume target queries. Breadth-first expansion under width/depth caps bounds total cost.
- **Access level**: Black-box (attacker LLM + target chat + evaluator/judge).
- **Key hyperparameters**: branching factor, tree width and depth caps, on-topic and
  harmfulness pruning thresholds, max total target queries.
- **Success signal**: evaluator harmfulness score crossing threshold on any tree node.
- **Citation**: Mehrotra, A., et al. (2024). *Tree of Attacks: Jailbreaking Black-Box LLMs
  Automatically.* NeurIPS 2024.
- **Pitfalls**: pruning thresholds are the crux — too aggressive discards promising branches,
  too lax negates the query-efficiency benefit over plain PAIR. Keep "on-topic" and "harmful"
  scoring genuinely separate (conflating them causes both false prunes and false survivals).
  Tree-state bookkeeping is materially more complex than PAIR's flat streams — test it directly.

### Many-shot Jailbreaking
- **Algorithm**: Construct one long synthetic conversation containing many (potentially
  hundreds of) fake user/assistant turns in which the assistant complies with requests that
  escalate from benign toward harmful, then append the real target request as the final turn —
  exploiting in-context learning over long context windows to shift behavior away from the
  model's trained refusal prior.
- **Access level**: Black-box; needs only a long-context chat interface (no gradients/weights),
  but the target's *maximum context length* is a binding constraint exposed via the model
  registry.
- **Key hyperparameters**: number of shots/turns, escalation gradient (how quickly fake turns
  ramp from benign to harmful), shot template/format, target context-window length.
- **Success signal**: this attack is largely a single constructed artifact rather than an
  iterative search — "success" is the judge verdict on the final turn's response. A shot-count
  sweep can be used to find the minimum number of shots that flips the verdict, which is the
  more informative reportable quantity than a single pass/fail.
- **Citation**: Anthropic (2024). *Many-shot Jailbreaking.*
  https://www.anthropic.com/research/many-shot-jailbreaking
- **Pitfalls**: effectiveness scales with context-window length — the model registry must
  expose this so the attack can size itself appropriately per target. Low-quality synthetic
  shots reduce potency; invest in shot-generation quality. Very long prompts are expensive at
  scale (token cost and latency) — budget accordingly, especially for shot-count sweeps.

### Crescendo
- **Algorithm**: A multi-turn conversation that starts on an innocuous, topically-related
  subject and escalates incrementally each turn — explicitly referencing the model's own prior
  (compliant) responses as context/justification for the next, slightly more harmful ask —
  until reaching the actual target behavior. Can be automated ("Crescendomation") with an
  attacker LLM planning the escalation path and handling backtracking when a turn is refused.
- **Access level**: Black-box (multi-turn chat; optionally an attacker LLM for automation).
- **Key hyperparameters**: number of escalation turns, escalation step size/aggressiveness,
  backtracking policy on refusal (retry / rephrase / step back).
- **Success signal**: judge verdict on the final turn's response; intermediate turns are judged
  for "still progressing toward target, not refused" to drive the continue/backtrack decision —
  a subtler, non-binary judgment than end-state jailbreak scoring.
- **Citation**: Russinovich, M., Salem, A., Eldan, R. (2024). *Great, Now Write an Article
  About That: The Crescendo Multi-Turn LLM Jailbreak Attack.* arXiv:2404.01833.
- **Pitfalls**: the conversation history *is* the attack state — model it explicitly and make
  it checkpointable/resumable. Backtracking logic is easy to get wrong (infinite retry loops, or
  giving up one step too early). Intermediate-turn judging needs a different rubric than
  end-state judging; don't reuse the final-verdict judge naively for in-progress turns.

### Best-of-N (BoN) Jailbreaking
- **Algorithm**: Apply random, semantics-preserving surface perturbations to the harmful prompt
  (random capitalization, character/word scrambling, punctuation/whitespace noise, synonym
  swaps — and, for multimodal targets, analogous perturbations to audio/image inputs) and
  sample N independent perturbed variants in parallel. Submit all N; the attack succeeds if
  *any* variant elicits a compliant response. No optimization or attacker model — pure random
  search exploiting the brittleness of safety training to surface-form noise.
- **Access level**: Black-box; purely a templating/sampling procedure plus chat access.
- **Key hyperparameters**: N (sample count), perturbation type mix and per-type probabilities,
  perturbation strength, early-stop-on-first-success toggle.
- **Success signal**: judge verdict on any of the N sampled responses — "success @ N" is the
  natural reportable curve, more informative than a single pass/fail at a fixed N.
- **Citation**: Hughes, J., et al. (2024). *Best-of-N Jailbreaking.* arXiv:2412.03556.
- **Pitfalls**: cost scales linearly with N — batch queries efficiently. Perturbation strength
  must stay within a range that preserves the prompt's meaning; too aggressive perturbation
  produces requests the target genuinely can't parse, which looks like (but isn't) a refusal —
  a false negative. Report success-rate curves over N, not just a final binary outcome.

### DeepInception
- **Algorithm**: Wrap the harmful request inside several nested layers of fictional framing
  (e.g. a story about characters inside a simulation, recursively explaining the request to
  characters in a deeper nested simulation), using the model's narrative/role-play capacity to
  diffuse responsibility for the content across fictional layers and slip past refusal
  heuristics tuned for direct requests. An attacker LLM can generate/vary the nested scenario.
- **Access level**: Black-box (a constructed prompt template; optionally an attacker LLM for
  scenario variation).
- **Key hyperparameters**: nesting depth, scenario/character templates, which nested layer
  carries the actual target instruction.
- **Success signal**: judge verdict on whether the innermost fictional "explanation" actually
  contains the harmful content (not just whether the model engaged with the fiction).
- **Citation**: Li, X., et al. (2023). *DeepInception: Hypnotize Large Language Model to Be
  Jailbreaker.* arXiv:2311.03191.
- **Pitfalls**: excessive nesting depth can produce incoherent output — a false negative that
  looks superficially like resistance but is actually confusion. Fixed scenario templates become
  pattern-matchable signatures for safety-tuned models — vary them. Judging requires looking
  *through* the fictional frame to the actual content, which is a different (harder) task than
  judging a direct response — make sure the judge prompt accounts for this.

### Cipher / Encoding-based Jailbreaks
- **Algorithm**: Encode the harmful instruction (and optionally require the response to be
  encoded too) using a transformation that the model can still parse but that evades
  keyword/embedding-based safety filters — character ciphers (ROT13, Caesar, Atbash), encodings
  (Base64, Morse, leetspeak), translation into low-resource languages, or structural/visual
  encodings (ASCII-art masking of trigger words, as in ArtPrompt). The model is instructed to
  decode, respond, and optionally re-encode.
- **Access level**: Black-box (templating plus chat access).
- **Key hyperparameters**: cipher/encoding type, whether the response must also be encoded,
  number and quality of few-shot decode examples included in the prompt (often necessary for
  reliable decoding of unfamiliar ciphers), and — for partial-encoding variants like
  ArtPrompt — which specific trigger word(s) get encoded versus left in plaintext.
- **Success signal**: judge verdict evaluated *after* decoding the (possibly encoded) response
  back to plaintext.
- **Citation**: Yuan, Y., et al. (2023). *GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs
  via Cipher.* arXiv:2308.06463. Jiang, F., et al. (2024). *ArtPrompt: ASCII Art-based Jailbreak
  Attacks against Aligned LLMs.* arXiv:2402.11753.
- **Pitfalls**: many models cannot reliably encode/decode obscure ciphers without few-shot
  examples — garbled output is a comprehension failure, not a refusal, and a naive judge will
  conflate the two. The mandatory decode step before judging is itself a place errors creep in.
  Effectiveness is strongly model- and tokenizer-dependent (varies with training-data language
  mix) — don't generalize results across model families without re-testing.

### GPTFuzzer
- **Algorithm**: A fuzzing loop seeded with a corpus of human-written jailbreak templates. Each
  iteration: select a seed (e.g. via an MCTS-style exploration/exploitation strategy), apply a
  mutation operator (an LLM rewrites/expands/shortens/crosses-over the template), substitute the
  target harmful question into the mutated template's placeholder, query the target, and score
  the response with a trained jailbreak classifier (the original work fine-tunes a RoBERTa
  model). Templates that succeed are folded back into the seed pool — explicitly measuring how
  well discovered templates generalize across many target questions.
- **Access level**: Black-box (mutator LLM + target chat + trained classifier judge).
- **Key hyperparameters**: seed corpus, seed-selection strategy (e.g. MCTS exploration
  constant), mutation-operator set and weights, query/iteration budget, classifier threshold.
- **Success signal**: the trained classifier's binary jailbreak verdict on the target response.
- **Citation**: Yu, J., et al. (2023). *GPTFUZZER: Red Teaming Large Language Models with
  Auto-Generated Jailbreak Prompts.* arXiv:2309.10253.
- **Pitfalls**: the in-loop classifier must be reasonably strong, or the fuzzer converges on
  templates that fool *the judge* rather than the target's actual safety training — a classic
  reward-hacking trap. Seed-corpus quality strongly biases the discovered template family. MCTS
  seed-energy/reward bookkeeping is easy to implement subtly incorrectly — test it in isolation.

---

## Gradient-based / white-box optimization attacks
*(owned by the gradient-attack-agent — `attacks/gradient_based/`)*

### Greedy Coordinate Gradient (GCG)
- **Algorithm**: Append a fixed-length adversarial suffix (initialized as repeated filler
  tokens) to the harmful prompt. Each step: compute the gradient of the loss — negative
  log-likelihood of an affirmative target string such as "Sure, here is how to…" — with respect
  to the one-hot encodings of the suffix tokens; for each suffix position take the top-k tokens
  by most-negative gradient as replacement candidates; randomly assemble B candidate suffixes
  by substituting one position each; batch-evaluate the *actual* loss for all B candidates with
  a forward pass (no gradient needed here); greedily keep the lowest-loss candidate; repeat for
  T steps. Can be made "universal/transferable" by averaging gradients and losses across
  multiple prompts and/or multiple models simultaneously.
- **Access level**: White-box — token-level logits *and* gradients with respect to one-hot
  input-token encodings (the standard "differentiate through the embedding lookup" trick).
- **Key hyperparameters**: suffix length, top-k candidate pool size, batch size B, number of
  optimization steps T, target-string template, and (for universal/transfer variants) the
  number of training prompts and models averaged over.
- **Success signal**: the optimization loss (negative log-likelihood of the target affirmative
  string) — convergence is "loss trending toward zero." This is an *in-loop* signal only;
  periodically sample real generations and pass them through the configured judges to confirm
  loss reduction is actually translating into jailbreak behavior, not just a degenerate output
  the loss function happens to favor.
- **Citation**: Zou, A., et al. (2023). *Universal and Transferable Adversarial Attacks on
  Aligned Language Models.* arXiv:2307.15043.
- **Pitfalls**: re-tokenizing `prompt + suffix + chat_template` as text can shift token
  boundaries relative to what was optimized, silently invalidating the suffix — decide
  explicitly whether to operate purely in token-ID space or to verify lossless text round-trips,
  and test for drift. Suffix placement relative to chat-template tokens must be identical
  between the optimization loop and the evaluation pathway. Per-step cost is dominated by the
  B-candidate batched forward pass — use KV-cache reuse on the shared prefix; an unoptimized
  inner loop makes everything downstream hard to iterate on.

### Embedding Attacks
- **Algorithm**: Rather than searching over discrete tokens (GCG), directly optimize a
  continuous vector — a "soft prompt" prepended or appended in embedding space — via standard
  gradient descent (e.g. Adam) on the same affirmative-target loss. The continuous search space
  converges faster and reaches lower loss than discrete search, making this useful as a
  white-box upper bound/diagnostic and for studying transferability — though the resulting
  embedding generally cannot be projected back to natural-language tokens for use against
  black-box/API targets.
- **Access level**: White-box — direct manipulation of input embeddings plus full
  backpropagation; the deepest input-side access level in the benchmark.
- **Key hyperparameters**: soft-prompt length, initialization (random vs. a real token
  sequence's embeddings), optimizer and learning rate, number of optimization steps, optional
  manifold-regularization terms that keep the embedding closer to the natural distribution.
- **Success signal**: the same affirmative-target loss as GCG, expected to converge faster and
  lower given the continuous search space.
- **Citation**: Schwinn, L., et al. (2024). *Soft Prompt Threats: Attacking Safety Alignment
  and Unlearning in Open-Source LLMs through the Embedding Space.* arXiv:2402.09063.
- **Pitfalls**: results do not transfer to black-box/API settings — frame and report this
  attack explicitly as a white-box ceiling/diagnostic, not a deployable jailbreak. Continuous
  optimization can drift embeddings far from the natural manifold, producing inputs that
  minimize loss while generating incoherent text — a reward-hacking failure mode that looks like
  convergence but isn't a real jailbreak; track embedding-norm drift and periodically validate
  with real generations. The same chat-template placement care as GCG applies.

---

## Representation-based / white-box activation attacks
*(owned by the representation-attack-agent — `attacks/representation_based/`)*

### Latent Perturbation
- **Algorithm**: Register forward hooks on one or more intermediate transformer layers. During
  generation, add a perturbation vector to the residual-stream/hidden-state activations at
  those layers — the vector can be random noise, a fixed steering vector, or itself optimized
  via gradient descent on the affirmative-target loss (mirroring GCG/embedding attacks, but in
  activation space rather than input space) — biasing generation away from refusal *without*
  modifying the input prompt at all.
- **Access level**: White-box — hidden-state read/write access via hooks (and gradients, if the
  perturbation vector itself is learned rather than fixed/random).
- **Key hyperparameters**: target layer(s) (mid-to-late layers are typically most relevant to
  refusal, but this is architecture-dependent and must be swept per model), perturbation type
  (random / fixed / optimized), magnitude or norm constraint, application scope (all token
  positions vs. only the generated continuation), and — if learned — the optimization
  hyperparameters.
- **Success signal**: if the perturbation is optimized, the affirmative-target loss serves as
  the in-loop signal; in all cases the attack's actual point is behavioral change, so the
  benchmark judges' verdicts on real generations are the metric that matters in the end.
- **Citation**: Representative of the activation-steering / latent-perturbation jailbreak
  family (e.g. Li, T., et al. (2024). *Open the Pandora's Box: Jailbreaking Open-Source LLMs
  through Logits and Activation Steering.*) — confirm the exact citation against whichever
  specific variant is implemented, and update this entry accordingly.
- **Pitfalls**: layer choice is the single most important and most architecture-dependent
  knob — treat it as a per-model sweep-and-validate step, never a hardcoded default. Excessive
  perturbation magnitude breaks output coherence wholesale, which superficially resembles (but
  is not) a refusal-bypass — a false positive for naive judges. Hook lifecycle management
  matters: an un-torn-down hook silently corrupts subsequent runs that share the same loaded
  model (which `models/loader.py` will do for efficiency).

### Refusal Direction Ablation
- **Algorithm**: (1) Run the model on matched sets of harmful and harmless instructions and
  extract residual-stream activations at each layer/token-position. (2) Compute the "refusal
  direction" as the difference-of-means vector between harmful- and harmless-instruction
  activations — empirically found to be a *single* direction that mediates refusal across many
  open-weight chat models. (3) "Ablate" it: at every layer, project this direction out of
  (subtract its component from) the residual stream at all token positions during generation.
  This projection is mathematically equivalent to zeroing the corresponding component of every
  weight matrix that writes into the residual stream — meaning the edit can be baked
  *permanently* into the weights with no fine-tuning at all. The complementary operation —
  *adding* the direction ("steering") — induces refusal on harmless inputs and is a useful
  validation check that the direction found is the right one.
- **Access level**: White-box — hidden-state extraction (to compute the direction) plus either
  runtime hooks or direct weight editing (to apply the ablation).
- **Key hyperparameters**: the harmful/harmless instruction sets used to compute the direction
  (and their size), candidate layer(s)/token-position(s) for extraction (selected via sweep
  against a held-out validation jailbreak rate, per the source paper — not fixed a priori),
  application mode (runtime ablation hook vs. permanent weight edit), and direction
  normalization.
- **Success signal**: validated two ways, both required: (a) bypass rate on a held-out harmful
  set, and (b) a no-regression check on a held-out harmless/benign set (refusal rate and
  output coherence/capability should remain essentially unchanged). The paper's central claim
  is *surgical precision* — reporting (a) without (b) misses the point entirely.
- **Citation**: Arditi, A., et al. (2024). *Refusal in Language Models Is Mediated by a Single
  Direction.* NeurIPS 2024 / arXiv:2406.11717.
- **Pitfalls**: direction extraction (which layer/position) is itself a small search problem
  with its own held-out validation set — never hardcode a single layer across models.
  Over-ablation degrades general capability; the benign-set regression check is not optional.
  Baking the edit into weights requires careful matrix algebra across every relevant weight
  matrix (a transposition or coverage mistake is easy to make silently) — get the hook-based
  runtime version working and validated *first*, then use it as ground truth to test the
  weight-edited version's equivalence against.

---

## Weight Tampering / white-box training attacks
*(owned by the weight-tampering-agent — `attacks/weight_tampering/`)*

### LoRA Fine-tuning
- **Algorithm**: Attach low-rank adapter matrices to the target model's attention/MLP
  projection layers and fine-tune *only the adapters* (base weights frozen) via standard
  supervised next-token-prediction loss on a small curated dataset of harmful-instruction →
  compliant-response pairs (the literature shows as few as ~10–100 examples can suffice),
  optionally mixed with benign examples to discourage wholesale capability collapse. After a
  small number of steps, the adapted model complies with requests the base model refused —
  demonstrating that RLHF/instruction-tuned safety alignment can be a thin, removable veneer
  rather than a robust property of the underlying weights.
- **Access level**: White-box — full weight access plus a training loop (forward, backward,
  optimizer step, checkpointing); the lightest-weight member of the weight-tampering family.
- **Key hyperparameters**: LoRA rank and alpha, target modules (which projections receive
  adapters), tampering-dataset size and harmful:benign composition ratio, learning rate,
  steps/epochs, optional capability-preservation regularization terms.
- **Success signal**: post-tampering compliance rate on held-out harmful prompts (via the
  standard judges), reported *together with* a capability-preservation measurement against (a)
  the pre-tampering base model and (b) a benign-capability benchmark — neither number means
  anything in isolation.
- **Citation**: Qi, X., et al. (2023/2024). *Fine-tuning Aligned Language Models Compromises
  Safety, Even When Users Do Not Intend To!* ICLR 2024 / arXiv:2310.03693.
- **Pitfalls**: the tampering dataset is the crux *and* a sensitive research artifact — keep it
  small, clearly labeled, version-controlled separately from the benchmark's evaluation dataset
  (never train on `dataset/socialharmbench.csv` itself — that would leak eval data into
  training and invalidate results), and documented for provenance. Too few steps under-tampers
  (residual refusals); too many over-tampers into incoherence. Compliance rate alone, without a
  capability-preservation number, is not a reportable result.

### Full-Parameter Fine-tuning
- **Algorithm**: Identical procedure to LoRA Fine-tuning — supervised fine-tuning on the same
  style of harmful-instruction → compliant-response dataset — but updating *all* model
  parameters rather than a low-rank adapter subset. This is the maximal/upper-bound version of
  the weight-tampering threat model, run primarily as a comparison point: how much (if
  anything) does the lightweight LoRA attack leave on the table relative to the strongest
  possible version of the same attack?
- **Access level**: White-box — full weight access plus a full training loop; substantially
  more compute and memory than LoRA (optimizer states for every parameter), typically requiring
  gradient checkpointing, mixed precision, and sharding (FSDP/ZeRO) to fit on available hardware.
- **Key hyperparameters**: the same dataset/training knobs as LoRA (size, composition, learning
  rate, steps/epochs) plus full-finetune-specific infrastructure knobs (precision, sharding
  strategy, gradient-accumulation steps, checkpoint frequency and retention).
- **Success signal**: identical reporting requirement to LoRA — compliance-rate change paired
  with capability preservation — with the added requirement that the comparison against LoRA be
  made at *matched dataset and (as feasible) compute budgets*; that comparison, not either
  attack's number in isolation, is the interesting research result.
- **Citation**: Qi, X., et al. (2023/2024) — same paper, comparing full fine-tuning as the
  stronger variant; broadly consistent with the wider "alignment is shallow" finding (cf.
  Lermen, S., et al. on LoRA vs. full fine-tuning for refusal removal).
- **Pitfalls**: the dominant pitfall here is infrastructure, not algorithm — budget
  memory/compute *before* attempting this beyond small models. Checkpoints are large; define
  and document a retention policy rather than accumulating them unmanaged. All the same
  dataset-handling-care and capability-preservation-reporting requirements from LoRA apply, and
  full fine-tuning is more prone to catastrophic forgetting, making the capability check even
  more load-bearing here than for LoRA.
