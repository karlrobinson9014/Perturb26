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
