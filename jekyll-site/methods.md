---
layout: default
title: Method
description: "Flow-OPD methodology: Flow-based Cold-Start, Multi-Teacher OPD, and Manifold Anchor Regularization"
permalink: /methods/
---

<div class="page-container">
    <div class="page-header">
        <div class="section-label">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><circle cx="12" cy="12" r="3"/></svg>
            Methodology
        </div>
        <h1>How Flow-OPD Works</h1>
    </div>

    <div class="content-section">
        <h2>Background: Flow Matching</h2>
        <p>
            Flow Matching (FM) maps a noise distribution \(p_0\) to data \(p_{\text{data}}\) via an ordinary differential equation (ODE):
            \(\text{d}\mathbf{x}_t = v_t(\mathbf{x}_t, t) \text{d}t\).
            Under the Optimal Transport (OT) formulation, the path is \(\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1\), and the model \(v_\theta\) learns the constant velocity \((\mathbf{x}_1 - \mathbf{x}_0)\) via the loss:
        </p>
        <div class="equation-box">
            \[\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1} \left[ \| v_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0) \|^2 \right]\]
            <span class="equation-label">Eq. 1: Flow Matching Loss</span>
        </div>
        <p>
            Following Flow-GRPO, we conceptualize the discretized ODE integration as a sequential <em>Markovian denoising process</em>, bridging continuous generative dynamics with reinforcement learning.
        </p>

        <h2>Motivation: Why GRPO Fails in Multi-Task Settings</h2>
        <p>
            Single-reward GRPO incurs severe <strong>catastrophic forgetting</strong> in orthogonal capabilities. This stems from <strong>unconstrained gradient interference</strong> driven by sparse scalar rewards within shared parameters \(\theta\).
        </p>
        <p>
            For a parameter update \(\Delta\theta\) driven by target task \(\mathcal{T}_1\), the collateral impact on an unmonitored capability \(\mathcal{T}_k\) can be approximated:
        </p>
        <div class="equation-box">
            \[\Delta \mathcal{J}_k \approx \langle \nabla_\theta \mathcal{J}_k, \Delta \theta \rangle \propto \mathbb{E}_{\mathbf{x} \sim \pi_\theta} \left[ A_1(\mathbf{x}) \left\langle \nabla_\theta \mathcal{J}_k, \nabla_\theta \log \pi_\theta(\mathbf{x} | c) \right\rangle \right]\]
            <span class="equation-label">Eq. 2: Gradient Interference</span>
        </div>
        <div class="insight-box">
            <h4>Key Insight</h4>
            <p>In high-dimensional spaces, divergent task gradients frequently conflict (\(\langle \nabla_\theta \mathcal{J}_k, \nabla_\theta \mathcal{J}_1 \rangle < 0\)). Mixing scalar rewards fails because it compresses multi-dimensional conflicts into a single advantage — a zero-sum game.</p>
        </div>
        <div class="insight-box">
            <h4>Our Solution</h4>
            <p>Replace sparse scalar rewards with <strong>dense, trajectory-level, multi-teacher supervision</strong>. Each teacher provides full-dimensional vector fields rather than single scalars, enabling uncoupled gradient signals for each capability.</p>
        </div>
    </div>

    <div class="content-section">
        <h2>Methodology: Flow-OPD Pipeline</h2>

        <div class="pipeline-diagram">
            <div class="pipeline-step">
                <div class="pipeline-step-number">1</div>
                <div class="pipeline-step-content">
                    <h3>Cold Start</h3>
                    <p>Establish a robust initial policy \(\theta_0\) using one of two strategies:</p>
                    <ul>
                        <li><strong>SFT-based:</strong> Fine-tune on trajectories sampled from specialized teachers.</li>
                        <li><strong>Model Merging:</strong> Superpose anisotropic priors of divergent teachers into a unified parameter state — no additional training needed.</li>
                    </ul>
                </div>
            </div>

            <div class="pipeline-step">
                <div class="pipeline-step-number">2</div>
                <div class="pipeline-step-content">
                    <h3>On-Policy Sampling</h3>
                    <p>The student explores the generative manifold by dynamically unrolling trajectories based on its current policy \(\pi_\theta\). We inject stochasticity by converting the deterministic ODE into an equivalent Stochastic Differential Equation (SDE):</p>
                    <div class="equation-box">
                        \[\text{d}x_t = \left[ v_{\theta}(x_t, t) + \frac{\sigma_t^2}{2t}(x_t + (1-t)v_{\theta}(x_t, t)) \right] \text{d}t + \sigma_t \text{d}w\]
                        <span class="equation-label">Eq. 3: SDE for On-Policy Exploration</span>
                    </div>
                    <p>This generates an on-policy marginal distribution \(x_t \sim \rho_t^\theta(\cdot | c)\), exposing intermediate states where the student is prone to hallucination.</p>
                </div>
            </div>

            <div class="pipeline-step">
                <div class="pipeline-step-number">3</div>
                <div class="pipeline-step-content">
                    <h3>Task-Specific Teacher Labeling</h3>
                    <p>At any explored state \(x_t\), the student queries an ensemble of expert teachers. Each teacher \(k\) acts as a <em>Generative Reward Model (GRM)</em>, providing a high-dimensional vector field \(v_{\phi_k}(x_t, t)\). A dynamic routing mechanism \(\alpha_k(c, x_t)\) activates experts based on the textual condition \(c\):</p>
                    <div class="equation-box">
                        \[v_{\text{target}}(x_t, t, c) = \sum_{k=1}^K \alpha_k(c, x_t) \, v_{\phi_k}(x_t, t, c)\]
                        <span class="equation-label">Eq. 4: Mode Blending of Teachers</span>
                    </div>
                </div>
            </div>

            <div class="pipeline-step">
                <div class="pipeline-step-number">4</div>
                <div class="pipeline-step-content">
                    <h3>Deriving Dense KL Reward</h3>
                    <p>Because both the student and target policies share the same isotropic covariance \(\sigma_t^2 \Delta t I\), their Reverse KL divergence reduces to:</p>
                    <div class="equation-box">
                        \[D_{\text{KL}}(\pi_\theta \| \pi_{\text{target}}) = \frac{\Delta t}{2} \left( \frac{\sigma_t(1-t)}{2t} + \frac{1}{\sigma_t} \right)^2 \| v_\theta - v_{\text{target}} \|^2\]
                        <span class="equation-label">Eq. 5: Dense KL Divergence as Vector Field Discrepancy</span>
                    </div>
                    <p>This provides a <strong>dense reward signal</strong> at every time step, replacing sparse scalar rewards.</p>
                </div>
            </div>

            <div class="pipeline-step">
                <div class="pipeline-step-number">5</div>
                <div class="pipeline-step-content">
                    <h3>Clipped Policy Gradient Update</h3>
                    <p>We incorporate a PPO-style clipping mechanism to stabilize training against high-frequency dense rewards:</p>
                    <div class="equation-box">
                        \[\mathcal{J}(\theta) \approx \frac{1}{B \times G} \sum_{j=1}^B \sum_{i=1}^G \sum_{t=0}^T \min \left( \rho_{t,i,j}(\theta) r_{t,i,j}^{\text{OPD}}, \; \text{clip}(\rho_{t,i,j}(\theta), 1-\epsilon, 1+\epsilon) r_{t,i,j}^{\text{OPD}} \right)\]
                        <span class="equation-label">Eq. 6: Clipped Surrogate Objective</span>
                    </div>
                    <p>Gradients flow exclusively through the policy ratio \(\rho_{t,i,j}(\theta)\), preserving fine-grained credit assignment while bounding the trust region.</p>
                </div>
            </div>
        </div>

        <h3>Manifold Anchor Regularization (MAR)</h3>
        <p>
            Aggressively optimizing functional targets often induces reward hacking, degrading visual aesthetics. We introduce a continuous-time aesthetic preservation mechanism: a frozen <em>aesthetic teacher</em> (e.g., DeQA-optimized) provides a regularizing vector field \(v_{\text{base}}\).
        </p>
        <div class="equation-box">
            \[\mathcal{L}_{\text{Total}}(\theta) = \mathcal{L}_{\text{Policy}}(\theta) + \lambda \, \mathbb{E}_{c, t, x_t \sim \rho_t^\theta} \left[ w(t) \| v_\theta(x_t, t, c) - v_{\text{aesthetic}}(x_t, t, c) \|^2 \right]\]
            <span class="equation-label">Eq. 7: Total Loss with Manifold Anchor Regularization</span>
        </div>
        <div class="insight-box">
            <h4>MAR Decouples Alignment from Aesthetics</h4>
            <p>While the student greedily absorbs functional intelligence from the multi-teacher ensemble, it remains strictly bounded to a high-quality visual manifold — completely averting the aesthetic degradation typical of single-objective RL.</p>
        </div>
    </div>
</div>
