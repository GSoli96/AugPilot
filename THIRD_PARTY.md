# Third-party code and data

AugPilot orchestrates algorithms written by other people. **This repository does not contain their code**:
`scripts/fetch_algorithms.py` clones each repository at a pinned commit (`scripts/algorithms_manifest.json`) into
`Algorithms/`, which is git-ignored. Each cloned repository stays under its own licence, and you are responsible for
respecting it. Licences were read from the `LICENSE` files and source headers of the clones used in the benchmark;
this is a summary, not legal advice.

| Licence of the clone | Repositories |
|---|---|
| MIT | GOGGLE, CasTGAN, TabFairGAN, CUTS, TabDiff, tab-ddpm and its CTGAN, timeVAE, diffusion-ts, TimeVQVAE, GraphENS, GraphCL, FLAG, GRAND, DropEdge, GAug, RGAN |
| Apache-2.0 | Fair-SMOTE, CTAB-GAN, TabSyn, Kamino, synthetic-tabular-LLM, tsgm, TSDiff, SimPSI, time_series_augmentation, TimeGAN, PyGCL |
| No licence file found | SMOTE-CDNN, PreFair, CTAB-GAN-Plus, OCTree, HARMONIC, KoVAE, Conditional-Sig-Wasserstein-GANs (SigCWGAN), GraphSmote, g-mixup |
| Restricted | DIG (GPL-3.0, copyleft), GraphAug (research evaluation only), ImagenTime (files under NVIDIA CC BY-NC-SA 4.0) |
| Not public | SyRFDRes (no public URL: ask its authors) |

Without a licence file, all rights stay with the authors of that repository: use it only for the research evaluation
it was published for, or ask the authors.

`scripts/patches/` holds our changes to MIT and Apache-2.0 repositories (CasTGAN, GOGGLE, TabFairGAN, TabDiff,
TabSyn); they keep the licence of the code they modify. Our changes to repositories without a licence file
(SMOTE-CDNN, tab-ddpm/CTAB-GAN-Plus) are not published, so `fetch_algorithms.py` reports those algorithms as failed to
patch and the affected methods need the unpatched code to be adapted by hand.

`pip` dependencies keep their own licences (see `envs_lock/`).

Datasets are not redistributed. Each comes from its original source (UCI, KEEL, TUDataset and others; see
`catalog_info/datasets_*.json` for the notes we kept) with its own terms, which we have not audited.
