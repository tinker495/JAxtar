# Applying Munchausen Reinforcement Learning to JAxtar TD Training

## Decision

The method in [arXiv:2007.14430](https://arxiv.org/abs/2007.14430) is **Munchausen
Reinforcement Learning (M-RL)**, not generic entropy regularization.  Its smallest
safe JAxtar integration is an opt-in `--munchausen` TD-target variant, defaulting to
off.  It must change only bootstrap TD targets; it must not change existing default
targets, optimizer, replay/sampling schedule, model architecture, or action-selection
policy.

JAxtar's public training documentation describes its Q-function as an expected
**cost**.  M-RL is written for reward maximization, so a cost-based target needs the
sign-converted form below; copying the paper's reward target unchanged into a
cost-minimizing update reverses the method's action preference.

## Primary sources and version context

- [Vieillard, Pietquin, and Geist, *Munchausen Reinforcement Learning*, NeurIPS
  2020, official paper PDF](https://proceedings.neurips.cc/paper_files/paper/2020/file/2c6a0bae0f071cbbf0bb3d5b11d90a82-Paper.pdf), especially §2, Eq. (2), pp. 2--3;
  §3, Eq. (3)--(4), pp. 3--4; and §4, pp. 5--7.
- [Official supplementary PDF](https://proceedings.neurips.cc/paper_files/paper/2020/file/2c6a0bae0f071cbbf0bb3d5b11d90a82-Supplemental.pdf), Appendix B.1, pp. 17--19:
  the complete M-DQN loss, Algorithm 1, numerical stabilization, and Table 2.
- [arXiv record, v3](https://arxiv.org/abs/2007.14430): submitted 2020-07-28;
  last revised 2020-11-04; NeurIPS 2020.  The paper itself links the authors'
  Google Research implementation.
- [van Seijen, Fatemi, and Tavakoli, *Using a Logarithmic Mapping to Enable
  Lower Discount Factors in Reinforcement Learning*, NeurIPS 2019, official
  paper PDF](https://proceedings.neurips.cc/paper_files/paper/2019/file/eba237eccc24353ccaa4d62013556ac6-Paper.pdf), §4, Eq. (5)--(6), pp. 4--5.
  This is a distinct primary source for transformed Bellman updates.  Its
  inverse map remains *inside* the ordinary discounted target; it does not
  justify adding an outer discount after taking a raw log-distance.
- First-party source-reference evidence is pinned to the initial author release,
  2020-07-30: [`google-research/google-research@72d82b6c56cc81e882be03c2620501df35829a9a:munchausen_rl/agents/m_dqn.py:L132-L174`](https://github.com/google-research/google-research/blob/72d82b6c56cc81e882be03c2620501df35829a9a/munchausen_rl/agents/m_dqn.py#L132-L174),
  [`...:common/utils.py:L30-L59`](https://github.com/google-research/google-research/blob/72d82b6c56cc81e882be03c2620501df35829a9a/munchausen_rl/common/utils.py#L30-L59), and
  [`...:train.py:L62-L65`](https://github.com/google-research/google-research/blob/72d82b6c56cc81e882be03c2620501df35829a9a/munchausen_rl/train.py#L62-L65).
  This source is supplemental to the paper, not a replacement for it.

## Exact method

For a target/frozen action-value vector \(\bar Q(s, \cdot)\) in a finite discrete
action space, M-RL derives a stochastic policy from the value function:

\[
\bar\pi(a\mid s) = \operatorname{softmax}(\bar Q(s,\cdot)/\tau)_a,
\qquad \tau > 0.
\]

For reward maximization, the one-step M-DQN regression target is

\[
y_R = r + \alpha\,[\tau\log\bar\pi(a\mid s)]_{l_0}^{0}
 + \gamma(1-d)\sum_{a'}\bar\pi(a'\mid s')
     [\bar Q(s',a')-\tau\log\bar\pi(a'\mid s')].
\]

Here \(d\) is the terminal indicator, \(\alpha\in[0,1]\), and \(l_0<0\) is the
log-policy floor.  The paper presents the unclipped equation as Eq. (2), then adds
the clipping rule in §4; the full clipped Huber regression loss is Supplement
Appendix B.1 Eq. (7), p. 17.  The immediate Munchausen term remains in a terminal
transition; only the bootstrapped continuation is multiplied by \(1-d\).  The
authors' implementation uses its target network for both \(\bar\pi(a\mid s)\) and
the \(s'\) soft backup, and stops gradients through the whole target
([M-DQN source, lines 137--165](https://github.com/google-research/google-research/blob/72d82b6c56cc81e882be03c2620501df35829a9a/munchausen_rl/agents/m_dqn.py#L137-L165)).

### Cost-minimizing JAxtar form

If JAxtar's TD quantity is a cost-to-go \(C=-Q_R\), set
\(\bar\pi(a\mid s)=\operatorname{softmax}(-\bar C(s,\cdot)/\tau)_a\).  Negating
the paper's reward target gives the target to minimize:

\[
y_C = c - \alpha\,[\tau\log\bar\pi(a\mid s)]_{l_0}^{0}
 + \gamma(1-d)\sum_{a'}\bar\pi(a'\mid s')
     [\bar C(s',a')+\tau\log\bar\pi(a'\mid s')].
\]

This is an algebraic sign conversion of paper Eq. (2), not a separately evaluated
variant.  It should be used only after confirming that the specific JAxtar label
path is indeed cost-minimizing; the public [Q-function training documentation](../qfunction_train.md)
uses that convention.

## Log-discounted terminal-reward distance: exact alignment

This section fixes a potentially confusing overloading in the preceding equations.
Use \(\rho\) for the **terminal-reward discount base** that defines a distance,
and \(\beta\) for the outer additive-RL discount in M-DQN.  They are not the
same operation after the logarithm.

### Deterministic shortest-path identity

For deterministic puzzle dynamics \(s_a=T(s,a)\), non-negative edge cost
\(c(s,a)\), and a solved state \(g\), define the ordinary lower-is-better
distance by

\[
D(g)=0,\qquad D(s)=\min_a\{c(s,a)+D(s_a)\}.
\]

Choose \(\rho\in(0,1)\), let \(\kappa=-\log\rho>0\), and give a solved state
the multiplicative terminal payoff \(U(g)=1\).  The corresponding discounted
terminal-reward value is

\[
U(s)=\max_a \rho^{c(s,a)}U(s_a)=\rho^{D(s)},\qquad
L(s)=\log U(s)=-\kappa D(s).
\]

Thus the exact log-space Bellman equation is

\[
L(g)=0,\qquad L(s)=\max_a\{c(s,a)\log\rho+L(s_a)\}.
\]

Equivalently, \(C=-L=\kappa D\) is a positive cost with
\(C(s)=\min_a\{\kappa c(s,a)+C(s_a)\}\).  Therefore **the log transform turns
the multiplicative terminal discount into the additive edge cost
\(-c\log\rho\); it does not leave a second outer factor \(\rho\) on the
continuation.**  Using
\(L\leftarrow c\log\rho+\rho\max L'\), or
\(D\leftarrow c+\rho\min D'\), does not equal \(\log(\rho^D)\) except in
degenerate cases.

For unit-cost moves and a model expressed directly in move-distance units, the
existing undiscounted JAxtar recurrence is the normalized choice
\(\kappa=1\), i.e. \(\rho=e^{-1}\).  An arbitrary \(\rho\) merely changes the
value unit from \(D\) to \(C=\kappa D\); it is not a continuation multiplier.

### Munchausen target in log and distance units

Apply paper Eq. (2) to the reward-side log value with \(\beta=1\):

\[
\begin{aligned}
\bar\pi(a\mid s)&=\operatorname{softmax}(\bar L(s,\cdot)/\tau_L)_a,\\
Y_L(s,a)&=c(s,a)\log\rho
 +\alpha[\tau_L\log\bar\pi(a\mid s)]_{l_{0,L}}^0\\
&\quad+(1-d_{s,a})\sum_b\bar\pi(b\mid s_a)
 [\bar L(s_a,b)-\tau_L\log\bar\pi(b\mid s_a)].
\end{aligned}
\]

Negating and dividing by \(\kappa\) retains lower-is-better model outputs
\(D_M=-L_M/\kappa\).  With

\[
\tau_D=\tau_L/\kappa,\qquad l_{0,D}=l_{0,L}/\kappa,
\]

the exactly equivalent distance-scale target is

\[
\begin{aligned}
\bar\pi(a\mid s)&=\operatorname{softmax}(-\bar D_M(s,\cdot)/\tau_D)_a,\\
Y_D(s,a)&=c(s,a)-\alpha[\tau_D\log\bar\pi(a\mid s)]_{l_{0,D}}^0\\
&\quad+(1-d_{s,a})\sum_b\bar\pi(b\mid s_a)
 [\bar D_M(s_a,b)+\tau_D\log\bar\pi(b\mid s_a)].
\end{aligned}
\]

This is precisely the cost-side sign conversion of M-RL Eq. (2), expressed in
distance units, with \(\beta=1\).  \(\alpha\) is dimensionless; temperature
and the log-policy floor have value units and **must** be rescaled by
\(1/\kappa\) when changing \(\rho\).  For example, treating the paper values
\(\tau_L=0.03,l_{0,L}=-1\) as log-return units gives
\(\tau_D\approx2.985,l_{0,D}\approx-99.5\) for \(\rho=0.99\), but exactly
\((0.03,-1)\) for the normalized \(\rho=e^{-1}\) convention.  Reusing
\((0.03,-1)\) after selecting \(\rho=0.99\) is a different algorithm, not a
unit-preserving translation of the paper setting.

The formula also gives the minimal code-level rule: when outputs remain move
distances, retain `softmax(-distance / tau_distance)`, unit edge costs, and no
outer discount in the continuation.  If a public `log_discount=rho` is added,
derive `tau_distance = tau_log / -log(rho)` and
`clip_distance = clip_log / -log(rho)`; do **not** multiply the continuation by
`rho`.

### What this does and does not prove

M-RL changes the action-value being fitted.  With \(\alpha>0\), including on a
transition into the goal, paper Eq. (2) retains the local \(\alpha\tau\log\pi\)
term and removes only the continuation.  Consequently \(D_M\) is a
lower-is-better **Munchausen/log-distance-scale proxy**, not the literal physical
distance \(D\) at finite \(\tau\) and \(\alpha\).  Masking that local term on
terminal edges would restore the literal terminal distance but would no longer be
the paper's target.  In particular, `--munchausen` must not claim heuristic
admissibility or exact path-length calibration; its value is action ordering and
the regularized TD objective.  The paper identifies M-VI with mirror-descent
value iteration having KL coefficient \(\alpha\tau\) and entropy coefficient
\((1-\alpha)\tau\) (paper §3, Theorem 1, p. 4), so for finite temperature this
is a regularized control objective rather than the unregularized shortest-path
one.

The paper's formal M-VI assumptions use an outer RL discount
\(\beta\in(0,1)\) (paper §2--3 and Supplement A, pp. 13--14).  Exact
\(L=\log(\rho^D)\) instead requires \(\beta=1\) in the additive log recursion.
For a finite acyclic/proper deterministic puzzle this recursion is well-defined,
but the paper's discounted-MDP contraction guarantee does not transfer merely by
setting \(\beta=1\).  If retaining that particular theorem is mandatory, apply
M-RL to an ordinary \(\beta<1\) return (or use the transformed update of van
Seijen et al., Eq. (5)); do not call the resulting value \(\log(\rho^D)\).

### Stochastic and terminal-convention boundaries

The shortest-path identity above relies on deterministic successors.  With
stochastic dynamics,

\[
L(s)=\max_a\left\{c(s,a)\log\rho+
\log\mathbb E_{s'\mid s,a}[\exp L(s')]\right\},
\]

not \(c\log\rho+\max_a\mathbb E[L(s')]\).  In distance units this is the
risk-sensitive backup

\[
D(s)=\min_a\left\{c(s,a)-\frac{1}{\kappa}
 \log\mathbb E_{s'\mid s,a}[\exp(-\kappa D(s'))]\right\}.
\]

An expected-distance TD target (including M-RL's action expectation) is therefore
not the log of an expected discounted terminal reward in a stochastic environment.
The equality holds for JAxtar's deterministic puzzle transition model, or only
conditionally on a realized deterministic successor.

There is also a one-step convention to fix explicitly.  If a conventional additive
return pays \(1\) **on the transition into** the goal and the goal is \(D\) edges
away, its return is \(\rho^{D-1}\), not \(\rho^D\).  To obtain \(\rho^D\), use
the boundary payoff \(U(g)=1\) after the final edge as above, pay \(\rho\) on
goal entry, or define distance with that extra terminal step.  JAxtar's
cost-to-go convention is the first formulation's clean boundary form:
\(D(g)=0\), the edge into the goal still contributes its cost, and the
continuation is zero.  Paper-consistent M-RL keeps the local Munchausen term on
that edge; only the continuation is terminal-masked.

## JAxtar-specific scalar heuristic / distance projection

### Evidence boundary

Everything through the action-cost target \(y_C(s,a)\) above is a direct algebraic
translation of the paper's Q-based M-DQN target (paper §2, Eq. (2), p. 3; Supplement
B.1 Eq. (7), p. 17).  The paper does **not** define a scalar state-value or
heuristic-only algorithm.  The projection in this section is therefore a JAxtar
integration inference: it reconstructs temporary action costs from a frozen scalar
distance model, applies the paper's Q target, and then reduces those action targets
back to one lower-is-better distance target.

### Induced action costs and policy

Let \(\bar V(s)\) be the frozen scalar model in **move-distance units**, let
\(s_a=T(s,a)\), and let \(c(s,a)\) be the transition cost.  Under the
log-discounted-distance convention above, form the paper's required complete
action-cost vector as

\[
\bar C(s,a) = c(s,a) + (1-d_{s,a})\bar V(s_a).
\]

The absent outer multiplier is intentional: it is the \(\beta=1\) additive
form of \(\log(\rho^D)\), not a choice of \(\rho=1\).  A conventional
discounted-MDP value may instead use
\(c+\beta(1-d)\bar V\), but then it is not the literal log-distance defined in
the preceding section.  Apply the valid-action mask before both normalization
operations, then define

\[
\bar\pi(a\mid s)=\operatorname{softmax}(-\bar C(s,\cdot)/\tau)_a,
\qquad
\ell(s,a)=[\tau\log\bar\pi(a\mid s)]_{l_0}^{0}.
\]

**`softmax(-distance / tau)` is the correct conversion.**  M-RL uses
`softmax(Q / tau)` for a reward-maximizing Q-value (paper §2, Eq. (1)--(2)); with
\(Q_R=-C\), its exact change of variables is `softmax(-C / tau)`.  It assigns more
probability to lower distances.  A reciprocal distance is neither derived by the
paper nor equivalent to the sign conversion; it is singular at zero and changes the
relative scale of every action.  Do not use `1 / distance` in this option.

### Projected target for a scalar distance model

For every valid action from \(s\), first compute the soft cost continuation at its
successor:

\[
\bar B(s_a)=\sum_{b\in A(s_a)}\bar\pi(b\mid s_a)
  [\bar C(s_a,b)+\ell(s_a,b)].
\]

The exact cost-side M-RL Q target reconstructed from \(\bar V\) is then

\[
Y_C(s,a)=c(s,a)-\alpha\ell(s,a)
 +(1-d_{s,a})\bar B(s_a).
\]

This is the \(\beta=1\) cost transform of Eq. (2), with \(\bar C(s,a)\) supplied
by the scalar model rather than an action-value head.  It needs **two-ply
successor enumeration**: all \(s_a\) to build the current policy and all
successors of every nonterminal \(s_a\) to build \(\bar B(s_a)\).  A scalar model
fed only sampled trajectory pairs cannot reproduce the paper's all-action soft
backup.

For a lower-distance heuristic whose existing semantics are
\(V(s)=\min_a C(s,a)\), the recommended projection is the greedy reduction

\[
Y_V(s)=\min_{a\in A(s)}Y_C(s,a).
\]

Train the online scalar model against `stop_gradient(Y_V)`.  This final `min` is an
explicit JAxtar-specific projection, not a result claimed by the paper; it preserves
the existing interpretation of a heuristic as the best available remaining cost.
If a caller instead wants the expected cost of the induced stochastic policy, it
may use \(\sum_a\bar\pi(a\mid s)Y_C(s,a)\), but that changes the value-model
semantics and is not the minimal heuristic integration.

For a terminal state, retain the established target \(V(s)=0\) (or its existing
terminal convention).  For a terminal **transition**, the current-action
Munchausen penalty remains in \(Y_C(s,a)\), while its continuation is zero, exactly
as in the paper's target-network implementation
([lines 152--165](https://github.com/google-research/google-research/blob/72d82b6c56cc81e882be03c2620501df35829a9a/munchausen_rl/agents/m_dqn.py#L152-L165)).

### Minimal heuristic implementation and test

Use the same default-off `--munchausen` switch.  In the scalar-heuristic TD branch
only, enumerate valid successors to create the temporary \(\bar C\) vectors,
compute `softmax(-C/tau)` and stable clipped `tau * log_pi`, form `Y_C`, and reduce
with `min`.  Preserve the current scalar TD target unchanged when the flag is off;
do not add a second public temperature or reciprocal-distance option.

The smallest meaningful test is a deterministic two-action, two-ply toy graph with
one terminal edge and one invalid action.  It must assert that: (1) lower induced
distance has higher `softmax(-C/tau)` probability; (2) the hand-computed `Y_C` and
`min(Y_C)` equal the implementation; (3) the terminal edge has no continuation but
does keep its local Munchausen term; (4) an invalid action has zero probability; and
(5) disabled mode equals the pre-existing scalar TD target exactly.

## What is required versus optional

| Item | Status | Evidence / consequence |
| --- | --- | --- |
| A full finite discrete-action Q/C vector and a valid-action mask, if the environment has invalid actions | Required | The policy and soft backup sum over the action set (paper §2, Eq. (1)--(2)).  Apply the *same* mask before both softmax/log-softmax computations; masked actions must contribute zero probability. |
| Frozen/target values for the entire target | Required | Eq. (2) uses \(q_{\bar\theta}\); the official code applies the target network at both \(s\) and \(s'\) ([lines 137--165](https://github.com/google-research/google-research/blob/72d82b6c56cc81e882be03c2620501df35829a9a/munchausen_rl/agents/m_dqn.py#L137-L165)). |
| Temperature-aware stable log-softmax plus clipping | Required for numerical safety | \(\log\pi\) is unbounded near a deterministic policy; the paper clips it (§4, p. 5) and Appendix B.1 derives a max-shifted temperature-aware log-sum-exp.  The author code implements exactly that ([`...:utils.py:L30-L59`](https://github.com/google-research/google-research/blob/72d82b6c56cc81e882be03c2620501df35829a9a/munchausen_rl/common/utils.py#L30-L59)). |
| Existing behavior/exploration policy | Preserve | M-DQN derives a stochastic policy for the target but the experiments retain epsilon-greedy interaction (§4, p. 5; Supplement B.2, pp. 19--20).  Do not make JAxtar behavior stochastic as part of this option. |
| Existing target-update, batch, and optimizer schedule | Preserve initially | The paper's DQN schedule is an experimental setting, not a M-RL requirement.  Its ablation shows changing RMSProp to Adam itself materially improves DQN (paper §4, p. 6), so changing it would confound the option's result. |
| Replay buffer / online data collection | Optional | Algorithm 1 uses FIFO replay, but the mathematical requirement is a TD batch and a frozen target.  Reuse JAxtar's existing TD sampling path. |
| IQN/distributional heads, n-step returns, PER, actor network | Not required | M-DQN is the minimal non-distributional result; M-IQN and 3-step returns are separate extensions (paper §2, p. 3; Supplement B.3). |

## Paper schedule and reference hyperparameters

For the Atari result only, the authors use one-step M-DQN, uniform FIFO replay,
batch 32, \(\gamma=0.99\), a gradient step every 4 environment frames, hard target
copy every 8,000 frames, a 1,000,000-transition buffer, and Adam with learning rate
\(5\times10^{-5}\) (Supplement B.1, Algorithm 1/Table 2, pp. 17--19).  Their
Munchausen parameters are \(\alpha=0.9\), \(\tau=0.03\), and \(l_0=-1\) (paper §4,
p. 5; Supplement Table 2).

Those values are a starting point, **not validated JAxtar defaults**: the paper tuned
them on a few Atari games with Atari reward scale and clipping.  In particular,
\(\tau\) has the same unit as Q/C values, so it must be calibrated against JAxtar's
distance/cost scale.  JAxtar uses the normalized log-distance coordinate
\(\kappa=-\log\rho=1\), so the private numeric triple `(0.9, 0.03, -1)` is in
that normalized coordinate.  Keep the public surface to the opt-in boolean first
and expose a different coordinate scale only if experiments require it.

## Ablation findings

- The essential intervention is the current-state Munchausen term, not merely a
  soft/maximum-entropy backup.  In the paper's ablation, M-DQN outperformed Adam
  DQN, both Soft-DQN comparisons, and Advantage Learning; Soft-DQN did not beat
  Adam DQN (§4, p. 6, Fig. 3).
- \(\alpha=0\) is Soft-DQN.  The implicit entropy coefficient for M-RL is
  \((1-\alpha)\tau\), so a fair Soft-DQN comparison uses that temperature, not
  necessarily \(\tau\) (paper §3, Theorem 1, p. 4; §4, p. 6).
- Large \(\alpha\) increases the action gap, but the theoretical \(\alpha\to1\)
  limit makes suboptimal values diverge; the paper explicitly recommends large but
  not too-close-to-one \(\alpha\) for numerical stability (Theorem 2, p. 5).
- Clipping is not an optional cosmetic change: it is the paper's response to the
  unbounded log-policy term (§4, p. 5).  Log-sum-exp stabilization is likewise part
  of its reference implementation (Supplement B.1, p. 18).

## Minimal opt-in implementation checklist

1. Add exactly one public training switch, `--munchausen`, default `False`.  It
   affects only bootstrap TD label construction; `diffusion` labels remain unchanged,
   and `warmup_td` uses it only after it enters its TD phase.
2. Leave the current target path byte-for-byte equivalent when the switch is false.
   Add a regression test asserting the default target is unchanged.
3. In the enabled branch, derive a masked \(\log\pi\) and \(\pi\) from the **frozen
   target** vector using stable, temperature-aware log-sum-exp.  Reuse the exact
   valid-action mask for current and next states.
4. Build the sign-correct target for the actual JAxtar convention.  For cost labels,
   use \(y_C\) above; retain terminal masking only on the continuation and apply
   `stop_gradient` to the completed target.
5. Use normalized log-distance scale \(\kappa=1\) with private constants
   `(alpha=0.9, tau_distance=0.03, log_pi_floor_distance=-1.0)`.  A different
   \(\rho\) must rescale both temperature and floor by \(1/(-\log\rho)\).
6. JAxtar intentionally min-caps enabled targets with physical diffusion distance.
   This preserves the existing overestimation guard but makes the implemented label
   a hybrid target rather than the pure M-RL operator above.
7. Add one focused test using a tiny two-action batch: verify (a) `--munchausen`
   false matches the old TD target, (b) an unlikely sampled action receives the
   expected cost-side penalty, (c) terminal samples have no continuation, and (d)
   masked actions receive no policy mass.
8. Evaluate baseline versus opt-in with identical seeds, labels, optimizer, target
   cadence, and sampling.  Tune \(\tau\) only after observing target/entropy scale;
   do not attribute gains to M-RL if any of those baseline settings changed.

## Reusable takeaway

Use the name **Munchausen Reinforcement Learning (M-RL)**.  For JAxtar, make it a
single default-off TD-target switch, preserve all existing training behavior, and
implement the cost-sign-converted, target-network, masked stable-log-softmax form.
The mandatory engineering pieces are stable \(\tau\log\pi\), clipping, terminal
handling, and a test that proves the disabled path is unchanged.
