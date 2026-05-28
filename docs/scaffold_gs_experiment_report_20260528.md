# Scaffold-GS Experiment Report: Error-Aware Refinement, Component Attribution, and View Encoding

Date: 2026-05-28

Repository branch: `experiment-summary-viewdist-pe-20260528`

Base branch before this report: `error-aware-anchor-refinement`

Upstream baseline reference: `origin/main` at `59c833b` (`Update README.md`)

Fork remote: `fengqiyu` (`https://github.com/fengqiyu0317/Scaffold-GS.git`)

## 1. Goal and Setup

This project studies Scaffold-GS from the machine-learning side of 3D Gaussian Splatting: learnable anchor / Gaussian representation, view-adaptive MLP prediction, differentiable rendering, photometric losses, Adam optimization, novel-view generalization, and density control as model-complexity management.

The experiments use the `truck` scene:

```text
data/tandt/truck
```

Unless noted otherwise, the common training configuration is:

```bash
python train.py --eval \
  -s data/tandt/truck \
  --gpu 0 \
  --voxel_size 0.01 \
  --update_init_factor 16 \
  --appearance_dim 0 \
  --ratio 1 \
  --iterations 30000
```

Metrics are computed on the test split. Higher SSIM / PSNR is better; lower LPIPS is better. FPS is useful for efficiency comparison, but runs executed concurrently are marked as not strictly comparable.

## 2. Code Changes Since Upstream

### 2.1 Error-aware anchor refinement

The first modification adds reconstruction-error statistics to the anchor refinement process. Instead of relying only on screen-space gradient and opacity statistics, the model accumulates rendering error on activated offsets / anchors and uses it during growing and pruning.

Main idea:

```text
rendered image vs. ground truth
-> pixel error / structure error / highlight error
-> project active neural Gaussians to image plane
-> accumulate error score per offset / anchor
-> adjust growing score and protect high-error anchors from pruning
```

This keeps the original RGB L1 + SSIM training loss intact while changing how limited model capacity is allocated.

### 2.2 Error-weighted photometric loss

The second modification changes the training objective by increasing the weight of hard image regions. It is a direct photometric-gradient intervention, complementary to error-aware density control.

The result improves PSNR strongly but does not beat error-aware refinement on LPIPS.

### 2.3 Component-aware refinement

The third line of experiments tries to use 2D component cues such as small highlights, thin white structures, local contrast, and vertical edge responses. The goal is to make small high-frequency structures influence anchor refinement more explicitly.

Implemented levels:

```text
level 0: reliable component attribution to existing visible offsets / anchors
level 1: level 0 + loose 2D proposal score around components
level 2: level 1 + ray-depth candidate proposal for missing anchors
```

Level 2 was also tested in a lighter configuration because the full proposal setting was slow:

```text
component_proposal_max_points=64
component_ray_depth_samples=1
component_candidate_max_per_interval=128
```

### 2.4 View / distance positional encoding

The current branch additionally includes direct Fourier positional encoding for Scaffold-GS view-adaptive MLP inputs.

New CLI arguments:

```text
--use_viewdist_pe
--view_pe_freqs
--dist_pe_freqs
--pe_include_input
```

The encoding can be applied to view direction and distance inputs used by color / opacity / covariance prediction branches. The experiments focus on whether higher input bandwidth helps the view-adaptive MLP recover high-frequency view-dependent details.

## 3. Main Results

| Method | Output directory | SSIM ↑ | PSNR ↑ | LPIPS ↓ | Test FPS ↑ | Size | Main conclusion |
|---|---|---:|---:|---:|---:|---:|---|
| Baseline Scaffold-GS | `final_30000` | 0.8817527 | 25.8097591 | 0.1434877 | 137.17 | 220M | Baseline |
| Error-aware refinement | `error_aware_30000_nohup_20260525_165859` | 0.8843653 | 25.9387188 | 0.1377494 | 126.74 | 254M | Best overall quality |
| Error-weighted loss | `error_aware_weighted_loss_30000_20260526_120555` | 0.8835869 | 25.9884472 | 0.1390346 | n/a | n/a | Best PSNR, LPIPS below error-aware |
| Component level 0 | `component_level0_ratio28_30000_` | 0.8784781 | 25.7200050 | 0.1470162 | n/a | n/a | Conservative but below baseline |
| Component level 1 | `component_level1_ratio28_30000_` | 0.8709760 | 25.4399815 | 0.1558293 | n/a | n/a | Loose proposal adds noise |
| Component level 2 light | `component_level2_light_ratio28_30000_20260528_172710` | 0.8750507 | 25.5382805 | 0.1499293 | n/a | n/a | Better than level 1, still below baseline |
| ViewDist PE 4/3 | `viewdist_pe_30000` | 0.8777 | 25.7754 | 0.1520 | 126.71 | 217M | High-frequency view+distance PE hurts |
| View PE3 | `view_pe3_30000` | 0.8790495 | 25.8689194 | 0.1491853 | 126.26 | 221M | PSNR rises, SSIM/LPIPS fall |
| View PE3 + Dist PE1 | `view_pe3_dist_pe1_30000` | 0.8784 | 25.7961 | 0.1497 | 125.15 | 213M | Distance PE hurts |
| View PE3 + error-aware | `view_pe3_error_aware_30000` | 0.8791695 | 25.7397499 | 0.1450265 | 108.46 | 286M | No complementarity with error-aware |
| View PE1 | `view_pe1_30000` | 0.8820678 | 25.8654747 | 0.1427840 | 55.49* | 215M | Most stable PE result |
| View PE2 | `view_pe2_30000` | 0.8817832 | 25.9161205 | 0.1441791 | 44.17* | 210M | Highest PSNR among view-only PE |
| View PE2 + Dist PE1 | `view_pe2_dist_pe1_30000` | 0.8808201 | 25.8125973 | 0.1448374 | 130.30 | 170M | Distance PE still degrades view-only result |

`*` View PE1 and View PE2 FPS were measured while two PE trainings / evaluations were running in parallel, so those FPS values should not be used for strict efficiency comparison.

## 4. Comparison Against Baseline

| Method | ΔSSIM | ΔPSNR | ΔLPIPS | Interpretation |
|---|---:|---:|---:|---|
| Error-aware refinement | +0.0026126 | +0.1289597 | -0.0057383 | Stable improvement on all metrics |
| Error-weighted loss | +0.0018342 | +0.1786881 | -0.0044531 | Strong PSNR gain, weaker perceptual gain |
| Component level 0 | -0.0032746 | -0.0897541 | +0.0035285 | Reliable attribution alone insufficient |
| Component level 1 | -0.0107767 | -0.3697776 | +0.0123416 | Loose proposal clearly harmful |
| Component level 2 light | -0.0067020 | -0.2714786 | +0.0064416 | Candidate proposal not yet effective |
| View PE1 | +0.0003151 | +0.0557156 | -0.0007038 | Small but consistent gain |
| View PE2 | +0.0000305 | +0.1063614 | +0.0006913 | PSNR gain with slight LPIPS regression |
| View PE2 + Dist PE1 | -0.0009326 | +0.0028381 | +0.0013497 | Distance PE removes the view-only benefit |

## 5. Interpretation by Experiment Family

### 5.1 Error-aware refinement is the strongest direction

Error-aware anchor refinement is the most defensible positive result. It improves SSIM, PSNR, and LPIPS simultaneously. The result supports the hypothesis that Scaffold-GS benefits from allocating anchor / Gaussian capacity according to reconstruction difficulty.

This is also easy to explain from a machine-learning perspective: density control acts as model-complexity management. High-error regions receive more representation capacity, while pruning is less likely to remove anchors that still explain hard pixels.

### 5.2 Error-weighted loss improves photometric fitting but is less balanced

The weighted photometric loss achieves the highest PSNR in the tested set. However, its LPIPS is worse than error-aware refinement. This suggests that directly increasing pixel-level gradient on high-error regions helps average reconstruction error, but it does not fully replace structural capacity allocation.

For the final course project, this is best presented as a complementary direction rather than the main method.

### 5.3 Component-aware refinement is a useful negative result

Component-aware refinement was motivated by observed failures on small bright points, thin rods, and local high-frequency structures. However, all tested component levels underperform the baseline.

The main failure reasons are:

- 2D component scores are hard to attribute to the correct 3D anchors.
- Loose proposal spreads signal to nearby background or unrelated structures.
- Level 2 candidate insertion was computationally expensive and, in the light run, did not successfully add useful candidate anchors.
- The method changes density-control signals, but not necessarily in geometrically correct 3D locations.

This should be reported as evidence that 2D structure detection alone is not enough. A stronger method would need multi-view consistency, better depth priors, or explicit candidate validation before anchor insertion.

### 5.4 View-only low-frequency PE has limited value

The early high-frequency PE experiments were negative: `view_pe_freqs=4`, `dist_pe_freqs=3`, and `view_pe_freqs=3` all degraded SSIM / LPIPS despite occasional PSNR improvements.

The later low-frequency experiments changed the picture:

- `View PE1` improves all three quality metrics slightly over baseline.
- `View PE2` gives the best PSNR among PE runs, but LPIPS is slightly worse than baseline.
- `View PE3` is worse than PE1 / PE2.

Therefore the correct conclusion is not that positional encoding is entirely useless. The more accurate conclusion is that low-frequency view-direction encoding can help, but the useful bandwidth is narrow.

### 5.5 Distance PE is not useful in the current branch

Both distance experiments are negative:

- `View PE3 + Dist PE1` is worse than `View PE3`.
- `View PE2 + Dist PE1` is worse than `View PE2` and slightly worse than baseline on SSIM / LPIPS.

This suggests that directly Fourier-encoding distance introduces unstable correlations into the color branch. Distance is a scale / geometry cue, not necessarily a periodic signal. Encoding it with sine and cosine features can increase MLP input bandwidth without adding reliable geometric information.

## 6. Local ROI Observations

Additional ROI checks on test view `00025.png` show that some PE variants improve selected thin white rod regions even when full-image metrics are worse. This is most visible for the `View PE3 + error-aware` run:

| ROI | Baseline MAE ↓ | View PE3 MAE ↓ | Error-aware MAE ↓ | View PE3 + error-aware MAE ↓ | Observation |
|---|---:|---:|---:|---:|---|
| Tree highlight region | 0.024780 | 0.023926 | 0.028319 | 0.022867 | ROI mean improves, but brightest pixels still weak |
| White rods full | 0.039280 | 0.037532 | 0.039644 | 0.034656 | PE + error-aware best locally |
| Left missing rod | 0.093117 | 0.090162 | 0.092502 | 0.086301 | PE + error-aware best locally |
| Right rod | 0.031395 | 0.030460 | 0.031691 | 0.027927 | PE + error-aware best locally |

These ROI gains are not enough to justify the PE + error-aware method as the main result because the full-image SSIM, PSNR, LPIPS, FPS, and model size all regress. They are still useful as qualitative evidence that view encoding can affect thin view-dependent highlights.

## 7. Final Ranking

By overall full-image quality, the current ranking is:

1. Error-aware anchor refinement
2. Error-weighted photometric loss
3. View PE1
4. View PE2
5. Baseline Scaffold-GS
6. View PE2 + Dist PE1
7. View PE3
8. Component level 0
9. Component level 2 light
10. Component level 1

The best main method remains error-aware anchor refinement. The best PE-only result is View PE1 if balanced quality matters, or View PE2 if PSNR is prioritized.

## 8. Recommended Report Framing

For the final course report, present the work as follows:

- Main positive result: error-aware anchor refinement improves novel-view rendering by using reconstruction error to guide density control.
- Supporting result: error-weighted loss confirms that hard pixels need stronger optimization signal, but loss weighting alone is less balanced than anchor-level capacity allocation.
- Negative but informative result: component-aware refinement shows that 2D small-structure cues require reliable 3D attribution; otherwise they can harm density control.
- PE result: low-frequency view-direction encoding gives small gains, while distance PE and high-frequency PE are not robust.

This framing keeps the project centered on machine-learning modeling rather than low-level CUDA or hardware optimization.

## 9. Future Work

Recommended next experiments:

1. Combine error-aware refinement with only `View PE1`, not PE3.
2. Re-evaluate View PE1 / View PE2 FPS serially if efficiency numbers are needed.
3. Add multi-view consistency checks for component proposals before they influence anchor growing.
4. Replace direct distance PE with a smoother learned distance embedding or normalized distance gate.
5. Test the best error-aware configuration on another scene to check generalization.
