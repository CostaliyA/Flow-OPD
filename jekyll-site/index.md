---
layout: default
title: Home
description: "Flow-OPD: On-Policy Distillation for Generalist Flow Matching Text-to-Image Generation — NeurIPS 2026"
---

<section class="hero">
    <div class="hero-inner">
        <div class="hero-badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
            NeurIPS 2026
        </div>

        <h1 class="hero-title">
            Flow-OPD: On-Policy Distillation for<br>
            <span>Flow Matching</span> Models
        </h1>

        <p class="hero-subtitle">
            The first unified post-training framework that integrates on-policy distillation into Flow Matching models,<br>
            effectively resolving reward sparsity and gradient interference in multi-task alignment.
        </p>

        <div class="hero-authors">
            <a href="mailto:fazii@mail.ustc.edu.cn">Zhen Fang</a><sup>*</sup> ·
            <a href="mailto:wxhuang@gmail.com">Wenxuan Huang</a><sup>*†</sup> ·
            Yu Zeng · Yiming Zhao · Shuang Chen ·
            Kaituo Feng · Yunlong Lin · Lin Chen ·
            Zehui Chen · Shaosheng Cao<sup>‡</sup> · Feng Zhao
        </div>

        <div class="hero-actions">
            <a href="https://github.com/CostaliyA/Flow-OPD" class="btn btn-primary" target="_blank" rel="noopener">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                GitHub Repository
            </a>
            <a href="{{ '/methods/' | relative_url }}" class="btn btn-outline">
                Read the Method
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
            </a>
        </div>

        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-stat-value">+29</div>
                <div class="hero-stat-label">GenEval Score ↑</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">+35</div>
                <div class="hero-stat-label">OCR Accuracy ↑</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">+10</div>
                <div class="hero-stat-label">pts over GRPO</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">4</div>
                <div class="hero-stat-label">Benchmarks</div>
            </div>
        </div>
    </div>
</section>

<section class="abstract-section">
    <div class="section-label">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
        Abstract
    </div>
    <div class="abstract-box">
        <p>
            Existing Flow Matching (FM) text-to-image models suffer from two critical bottlenecks under multi-task alignment: the <strong>reward sparsity</strong> induced by scalar-valued rewards, and the <strong>gradient interference</strong> arising from jointly optimizing heterogeneous objectives, which together give rise to a <em>"seesaw effect"</em> of competing metrics and pervasive reward hacking. Inspired by the success of On-Policy Distillation (OPD) in the large language model community, we propose <strong>Flow-OPD</strong>, the first unified post-training framework that integrates on-policy distillation into Flow Matching models. Flow-OPD adopts a two-stage alignment strategy: it first cultivates domain-specialized teacher models via single-reward GRPO fine-tuning, allowing each expert to reach its performance ceiling in isolation; it then establishes a robust initial policy through a Flow-based Cold-Start scheme and seamlessly consolidates heterogeneous expertise into a single student via a three-step orchestration of on-policy sampling, task-routing labeling, and dense trajectory-level supervision. We further introduce <strong>Manifold Anchor Regularization (MAR)</strong>, which leverages a task-agnostic teacher to provide full-data supervision that anchors generation to a high-quality manifold, effectively mitigating the aesthetic degradation commonly observed in purely RL-driven alignment. Built upon Stable Diffusion 3.5 Medium, Flow-OPD raises the GenEval score from 63 to 92 and the OCR accuracy from 59 to 94, yielding an overall improvement of roughly <strong>10 points over vanilla GRPO</strong>, while preserving image fidelity and human-preference alignment and exhibiting an emergent <strong>"teacher-surpassing" effect</strong>. These results establish Flow-OPD as a scalable alignment paradigm for building generalist text-to-image models.
        </p>
    </div>
</section>

<section class="features-section">
    <div class="section-label">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
        Key Contributions
    </div>
    <div class="features-grid">
        <div class="feature-card">
            <div class="feature-icon purple">🔬</div>
            <h3>Analysis of Multi-task FM Training</h3>
            <p>We provide an empirical analysis of the failure modes of GRPO-based multi-task training in Flow Matching models, specifically identifying reward sparsity and gradient interference as the core bottlenecks.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon cyan">⚙️</div>
            <h3>The Flow-OPD Framework</h3>
            <p>A two-stage post-training framework that decouples expertise acquisition from model unification. Introduces Flow-based Cold-Start (SFT & Merging), task routing dense labeling, and Manifold Anchor Regularization.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon green">🏆</div>
            <h3>Superior Performance</h3>
            <p>Substantial 10-point improvement over GRPO baseline across four mainstream benchmarks. The unified student matches or surpasses specialized teachers in-domain with exceptional OOD generalization.</p>
        </div>
    </div>
</section>
