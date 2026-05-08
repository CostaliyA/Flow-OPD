---
layout: default
title: Results
description: "Flow-OPD quantitative results on GenEval, OCR, DeQA, PickScore, and T2I-CompBench"
permalink: /results/
---

<div class="page-container">
    <div class="page-header">
        <div class="section-label">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
            Main Results
        </div>
        <h1>Experimental Results</h1>
    </div>

    <div class="content-section">
        <h2>Main Performance Comparison</h2>
        <p>
            Flow-OPD consistently matches or surpasses specialized teacher models across all benchmarks. It resolves severe cross-domain interference inherent to specialization (e.g., PickScore teacher's GenEval drops to 0.51) and overcomes the optimization bottlenecks of sparse-reward multi-task GRPO.
        </p>

        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>GenEval ↑</th>
                        <th>OCR Acc. ↑</th>
                        <th>DeQA ↑</th>
                        <th>PickScore ↑</th>
                        <th>Avg ↑</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>SD-3.5-M</strong> (base)</td>
                        <td>0.63</td>
                        <td>0.59</td>
                        <td>4.07</td>
                        <td>21.64</td>
                        <td>0.7166</td>
                    </tr>
                    <tr>
                        <td colspan="6" style="background: var(--color-surface-alt); font-weight:600; font-size:0.8rem; color: var(--color-text-muted);">Single-Reward GRPO Teachers</td>
                    </tr>
                    <tr>
                        <td>+ GRPO-GenEval</td>
                        <td class="td-bold">0.94</td>
                        <td>0.65</td>
                        <td>4.01</td>
                        <td>21.53</td>
                        <td>0.8050</td>
                    </tr>
                    <tr>
                        <td>+ GRPO-OCR</td>
                        <td>0.64</td>
                        <td class="td-bold">0.92</td>
                        <td>4.06</td>
                        <td>21.69</td>
                        <td>0.8016</td>
                    </tr>
                    <tr>
                        <td>+ GRPO-DeQA</td>
                        <td>0.64</td>
                        <td>0.66</td>
                        <td class="td-bold">4.23</td>
                        <td class="td-second">23.02</td>
                        <td>0.7578</td>
                    </tr>
                    <tr>
                        <td>+ GRPO-PickScore</td>
                        <td class="td-drop">0.51</td>
                        <td>0.69</td>
                        <td>4.22</td>
                        <td class="td-bold">23.19</td>
                        <td>0.7340</td>
                    </tr>
                    <tr>
                        <td colspan="6" style="background: var(--color-surface-alt); font-weight:600; font-size:0.8rem; color: var(--color-text-muted);">Multi-Reward GRPO Baselines</td>
                    </tr>
                    <tr>
                        <td>GRPO-Mix</td>
                        <td>0.73</td>
                        <td>0.83</td>
                        <td class="td-second">4.33</td>
                        <td>21.84</td>
                        <td>0.8165</td>
                    </tr>
                    <tr>
                        <td>SFT + GRPO-Mix</td>
                        <td>0.85</td>
                        <td>0.86</td>
                        <td>4.29</td>
                        <td>21.79</td>
                        <td>—</td>
                    </tr>
                    <tr>
                        <td>Merge + GRPO-Mix</td>
                        <td>0.84</td>
                        <td>0.86</td>
                        <td>4.18</td>
                        <td>21.87</td>
                        <td>—</td>
                    </tr>
                    <tr>
                        <td colspan="6" style="background: var(--color-surface-alt); font-weight:600; font-size:0.8rem; color: var(--color-text-muted);">Flow-OPD (Ours)</td>
                    </tr>
                    <tr>
                        <td><strong>Ours (SFT)</strong></td>
                        <td class="td-second">0.91</td>
                        <td class="td-second">0.92</td>
                        <td>4.29</td>
                        <td>21.83</td>
                        <td class="td-second">0.8819</td>
                    </tr>
                    <tr>
                        <td><strong>Ours (Merge)</strong></td>
                        <td class="td-best">0.92</td>
                        <td class="td-best">0.94</td>
                        <td class="td-best">4.35</td>
                        <td class="td-best">23.08</td>
                        <td class="td-best">0.9044</td>
                    </tr>
                </tbody>
            </table>
        </div>
        <p class="table-note">
            ↑ Higher is better for all metrics. Scores of teacher models are <strong>bolded</strong> to denote performance ceilings. Blue cells indicate the best scores; green cells indicate the second best. GRPO-Mix averages are computed by averaging four 0-1 normalized metrics.
        </p>

        <div class="insight-box">
            <h4>Teacher-Surpassing Effect</h4>
            <p>Flow-OPD excels in certain edge cases where all individual teachers fail. We hypothesize this emergent superiority stems from <em>knowledge cross-pollination</em> within the latent flow manifold. While individual teachers are constrained by domain-specific biases, simultaneous dense guidance forces the student to learn a more holistic, smoothed representation.</p>
        </div>

        <h2>Image Quality &amp; Human Preference Metrics</h2>
        <p>Comparison on general image quality and alignment metrics beyond the core benchmarks.</p>

        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>ImageReward ↑</th>
                        <th>Aesthetic ↑</th>
                        <th>UnifiedReward ↑</th>
                        <th>HPS-v2.1 ↑</th>
                        <th>QwenVL ↑</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>SD-3.5-M</td>
                        <td>1.02</td>
                        <td>5.87</td>
                        <td>3.339</td>
                        <td>0.2982</td>
                        <td>3.45</td>
                    </tr>
                    <tr>
                        <td>GRPO-DeQA</td>
                        <td>1.33</td>
                        <td>5.97</td>
                        <td>3.456</td>
                        <td>0.2846</td>
                        <td>3.68</td>
                    </tr>
                    <tr>
                        <td>GRPO-Mix</td>
                        <td>1.23</td>
                        <td>5.93</td>
                        <td>3.501</td>
                        <td>0.3101</td>
                        <td>3.88</td>
                    </tr>
                    <tr>
                        <td>w/o. MAR</td>
                        <td>1.26</td>
                        <td>5.89</td>
                        <td>3.518</td>
                        <td>0.2998</td>
                        <td>3.82</td>
                    </tr>
                    <tr>
                        <td><strong>Ours (Merge)</strong></td>
                        <td class="td-best">1.36</td>
                        <td class="td-best">6.23</td>
                        <td class="td-best">3.659</td>
                        <td class="td-best">0.3302</td>
                        <td class="td-best">4.05</td>
                    </tr>
                </tbody>
            </table>
        </div>
        <p class="table-note">
            MAR (Manifold Anchor Regularization) provides full-data supervision across the entire dataset, significantly enhancing both visual quality and expressive power of generated images.
        </p>

        <h2>T2I-CompBench++ (OOD Generalization)</h2>
        <p>Out-of-distribution evaluation on T2I-CompBench++ to assess compositional generalization capabilities.</p>

        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Color</th>
                        <th>Shape</th>
                        <th>Texture</th>
                        <th>Complex</th>
                        <th>3D-Spatial</th>
                        <th>Numeracy</th>
                        <th>Non-Spatial</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>SD-3.5-M</td>
                        <td>0.799</td>
                        <td>0.567</td>
                        <td>0.734</td>
                        <td>0.380</td>
                        <td>0.374</td>
                        <td>0.593</td>
                        <td>0.315</td>
                    </tr>
                    <tr>
                        <td>GRPO-Mix</td>
                        <td>0.797</td>
                        <td>0.580</td>
                        <td>0.739</td>
                        <td>0.368</td>
                        <td>0.368</td>
                        <td>0.639</td>
                        <td>0.313</td>
                    </tr>
                    <tr>
                        <td>Cold Start</td>
                        <td>0.817</td>
                        <td>0.613</td>
                        <td>0.734</td>
                        <td>0.387</td>
                        <td>0.425</td>
                        <td>0.646</td>
                        <td>0.315</td>
                    </tr>
                    <tr>
                        <td>Cold Start + GRPO</td>
                        <td>0.803</td>
                        <td>0.599</td>
                        <td>0.741</td>
                        <td>0.384</td>
                        <td>0.402</td>
                        <td>0.627</td>
                        <td>0.314</td>
                    </tr>
                    <tr>
                        <td><strong>Ours (Merge)</strong></td>
                        <td class="td-best">0.830</td>
                        <td class="td-best">0.629</td>
                        <td class="td-best">0.745</td>
                        <td class="td-best">0.394</td>
                        <td class="td-best">0.457</td>
                        <td class="td-best">0.684</td>
                        <td class="td-best">0.316</td>
                    </tr>
                </tbody>
            </table>
        </div>
        <p class="table-note">
            Flow-OPD achieves state-of-the-art performance across all seven compositional dimensions. Note that Cold Start + GRPO suffers from catastrophic forgetting (e.g., shape 0.599 vs. Cold Start 0.613), while Flow-OPD effectively mitigates these regressions.
        </p>

        <h2>Multi-Reward Gradient Interference</h2>
        <p>Sequential stacking of rewards demonstrates the "seesaw effect" in multi-reward optimization — each added reward causes catastrophic forgetting of previously learned capabilities.</p>

        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>GenEval ↑</th>
                        <th>OCR Acc. ↑</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>SD-3.5-M</td>
                        <td>0.63</td>
                        <td>0.59</td>
                    </tr>
                    <tr>
                        <td>+ GenEval</td>
                        <td class="td-bold">0.94</td>
                        <td>0.65</td>
                    </tr>
                    <tr>
                        <td>+ OCR</td>
                        <td class="td-drop">0.89 <span style="color:var(--color-danger); font-size:0.75rem;">(↓5%)</span></td>
                        <td class="td-bold">0.91</td>
                    </tr>
                    <tr>
                        <td>+ PickScore</td>
                        <td class="td-drop">0.82 <span style="color:var(--color-danger); font-size:0.75rem;">(↓7%)</span></td>
                        <td class="td-drop">0.86 <span style="color:var(--color-danger); font-size:0.75rem;">(↓5%)</span></td>
                    </tr>
                    <tr>
                        <td>+ DeQA</td>
                        <td class="td-drop">0.73 <span style="color:var(--color-danger); font-size:0.75rem;">(↓9%)</span></td>
                        <td class="td-drop">0.83 <span style="color:var(--color-danger); font-size:0.75rem;">(↓3%)</span></td>
                    </tr>
                </tbody>
            </table>
        </div>
        <p class="table-note">
            Each additional reward causes severe capability degradation, confirming gradient interference (\(\langle \nabla_\theta \mathcal{J}_i, \nabla_\theta \mathcal{J}_j \rangle < 0\)). Scalar reward mixing is fundamentally unscalable due to the sparse information bottleneck.
        </p>

        <h2>Comparison with DiffusionNFT</h2>

        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Qwen-VL Score ↑</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>DiffusionNFT</td>
                        <td>3.74</td>
                    </tr>
                    <tr>
                        <td><strong>Flow-OPD (Ours)</strong></td>
                        <td class="td-best">4.05</td>
                    </tr>
                </tbody>
            </table>
        </div>
        <p class="table-note">
            Flow-OPD significantly outperforms DiffusionNFT (+0.31) in human preference alignment as evaluated by Qwen-VL. DiffusionNFT's incompatibility with CFG and pronounced reward hacking behaviors are the main causes of its lower scores.
        </p>
    </div>
</div>
