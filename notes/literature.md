# Literature (research mode: search Consensus / SciSpace, record here, read before searching)

One bullet per paper: title (year), link, the one idea that applies to Perturb, tried? (iter N, outcome).

- Sparse-RS: a versatile framework for query-efficient sparse black-box adversarial attacks (Croce et al., 2020),
  https://consensus.app/papers/details/4ee5a69568e451f39552aa86962b3a42/ : random search over sparse supports;
  black-box, but its support-resampling moves could replace ISTA's support updates. Not tried.
- Group-wise Sparse and Explainable Adversarial Attacks (Sadiku, Wagner, Pokutta, 2023): group sparsity via
  quasinorm; our sparse-group-lasso test (research_log 2026-10-19) clustered changes without reducing their count.
- MALT: Mesoscopic Almost Linearity Targeting (Melamed, Yehudai, Shamir, 2024), https://arxiv.org/pdf/2407.02240 :
  picks target classes by a linear estimate (logit gap / gradient norm) instead of raw confidence. Our own linear
  screen (research_log 2026-10-19) ranked the competitors' classes 129-605th, so on this model the cheap classes are
  nonlinear; not tried as such (iter 1 uses discovered classes instead).
- Analysis of Dominant Classes in Universal Adversarial Perturbations (Vadillo et al., 2020),
  https://consensus.app/papers/details/88115783465050c887a573a92db89ad2/ and Data-free UAP (Zhang et al., ICCV 2021),
  https://consensus.app/papers/details/ef2ff251f180572da954500a2fe87f2f/ : untargeted universal perturbations send most
  images to one dominant class, suggesting a global class prior. Checked iter 1 on dev competitors
  (`tools/competitor_classes.py`, results/competitor-classes-dev.json): no dominant class (most common best class 151,
  4/64 images), so a global prior is weak for our sparse per-image setting.
- Evaluating Adversarial Attacks on ImageNet: A Reality Check on Misclassification Classes (Ozbulak et al., 2021),
  https://consensus.app/papers/details/80418f69e5fa5c388bc09ab6e8fae6e1/ : 71 % of transferable adversarials land in the
  clean top-5. Opposite here: the competitors' best classes have clean rank >= 20 for most images (median ~90).
- Sparse and Imperceivable Adversarial Attacks (Croce, Hein, ICCV 2019),
  https://consensus.app/papers/details/45fa7e542366588ead9e6f85782c8a21/ and Probabilistic JSMA (Combey et al., 2020),
  https://consensus.app/papers/details/2620d29474a953528dd1e0ae30af6795/ : L0 attacks; JSMA-style targeted saliency is
  the classic way to run an attack per target class. Idea: race several targets and keep the cheapest (iter 1).
- Multi-attacks (Fort, 2023), https://arxiv.org/abs/2308.03792 : one image's neighbourhood holds ~10^O(100) class regions,
  i.e. nearly any class is reachable; supports "the class is a free choice, the cost differs".
- Enhancing Sparsity by Reweighted L1 Minimization (Candes, Wakin, Boyd, 2008), https://doi.org/10.1007/s00041-008-9045-x :
  per-element L1 weights 1/(|x|+eps) from the previous iterate approximate L0. Tried iter 205 as `rw_eps` in the ISTA
  threshold: worse (16-img -0.0014..-0.0023, more nonzeros); the inverse weighting (saturated entries shrink harder) won.
- Between hard and soft thresholding: optimal iterative thresholding algorithms (Liu, Barber, 2018),
  https://consensus.app/papers/details/e74b7313534d5dd6b7c83d167af53c24/ : thresholding operators between soft and hard
  (lq, reciprocal thresholding) have better guarantees than either. Here: a smoothed-step shrink profile at the 0.5
  rounding point (`rw_step`), tried iter 207 (see research_log).
- sigma-zero: Gradient-based Optimization of l0-norm Adversarial Examples (Cina et al., 2024),
  https://consensus.app/papers/details/16ff8bc344305450b4853b10a634b1e2/ : smooth l0 surrogate x^2/(x^2+sigma^2) plus an
  adaptive threshold projection. Its gradient is largest for small entries (like reweighted L1, which lost in iter 205),
  so not tried as such; its adaptive threshold matches our lam schedule.
- A New Proximal Iterative Hard Thresholding Method with Extrapolation (Zhang et al., 2018),
  https://consensus.app/papers/details/80f75b58ab3d51e293a3d37febfa9602/ and Accelerated IHT for l0 box-constrained
  problems (Wu et al., 2019), https://consensus.app/papers/details/14ffc81b82965c2780eae987d5b9b939/ : l0 + box via
  proximal hard thresholding with extrapolation (momentum on d). Momentum on the ISTA iterate not tried yet (ideas.md).
- Sparse and Imperceptible Adversarial Attack via a Homotopy Algorithm (Zhu, Chen, Wang, 2021),
  https://arxiv.org/abs/2106.06027 : l0-regularised loss with nmAPG, then an "l0 change control" step that limits how
  many entries enter/leave per iteration, and a post-attack step to escape bad minima. Not tried.
- Fast Minimum-norm Adversarial Attacks through Adaptive Norm Constraints (Pintor et al., 2021),
  https://consensus.app/papers/details/1d2773a56d0359d5b9a67ba771dd54e4/ : adapts the norm budget eps up/down around the
  boundary, like our lam_up/lam_down; HO-FMN (Mura et al., 2024) finds step-size schedulers matter. Motivates the
  iter 207 lr/lam retune.
- EAD: Elastic-Net Attacks to Deep Neural Networks via Adversarial Examples (Chen et al., 2017),
  https://consensus.app/papers/details/eb8f374526375015943c139395bac649/ : L1 (elastic-net) attack solved with
  ISTA/FISTA, i.e. our lane update plus Nesterov extrapolation. Tried iter 208 as `extrap` (see research_log).
- Adaptive Restart for Accelerated Gradient Schemes (O'Donoghue, Candes, 2015), https://arxiv.org/abs/1204.3982 and
  Proximal Gradient Algorithm with Momentum and Flexible Parameter Restart for Nonconvex Optimization (Zhou et al., 2020),
  https://arxiv.org/abs/2002.11582 : momentum in proximal gradient needs a restart when the iterates oscillate; for us,
  restart on re-target (done in iter 208); an oscillation-triggered restart not tried.
- Adaptive FISTA for Non-convex Optimization (Ochs, Pock, 2017), https://arxiv.org/abs/1711.04343 : line-searched
  extrapolation parameter; too costly per step here (each objective eval is a forward pass). Not tried.
- Generate Adversarial Examples by Nesterov-momentum Iterative FGSM (Xu, 2020), doi 10.1109/ICSESS49938.2020.9237700,
  and On the Convergence of an Adaptive Momentum Method for Adversarial Attacks (Sheng et al., AAAI 2024),
  doi 10.1609/aaai.v38i13.29323 : gradient-EMA momentum in iterative attacks gives smaller perturbations at the same
  success rate. Tried iter 208 as `momentum` (EMA of the normalised gradient, see research_log).
- The Power of Decaying Steps: Enhancing Attack Stability and Transferability for Sign-based Optimizers (Tao et al., 2026),
  https://consensus.app/papers/details/d7e6dd39da665f7fa420039cf5874e4c/ : non-decaying step sizes make iterative attacks
  non-convergent / unstable; monotonically decreasing coordinate-wise steps (MDCS-MI, with momentum) converge at O(1/sqrt T).
  Here: per-lane linear lr decay `lr_final` on top of momentum, tried iter 209 (see research_log).
- Boosting Black-box Adversarial Attack with a Better Convergence (Yin et al., 2020),
  https://consensus.app/papers/details/915aaa7a246d563aa4216b58df3a004b/ : step-size decay in (M)I-FGSM gives better
  convergence. Same idea as above; iter 209.
- Non-Lipschitz Attack: a more sparse adversarial attack via non-Lipschitz lp regularization (Lin, Wu, Xiang, 2023),
  doi 10.4208/csiam-am.so-2022-0005 : lp (0<p<1) model with a support lower bound -> iterative support *shrinking*
  (entries that leave the support never re-enter) plus thresholding; ~14 % fewer pixels than FMN-l0. Not tried;
  candidate for a late-budget "freeze support" phase.
- Efficient Untargeted White-Box Adversarial Attacks Based on Simple Initialization (Zhou et al., 2024),
  https://consensus.app/papers/details/857dd1fa722956ccb112e7a968c935d3/ : a guided init instead of random restarts.
  Relevant to warm-starting re-targeted lanes (they restart from d=0 at 35 % of the budget). Not tried.
