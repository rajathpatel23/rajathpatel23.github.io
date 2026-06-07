---
title: "Projects"
date: 2025-01-27
draft: false
showToc: false
TocOpen: false
---

<style>
/* ---- Break out of theme's 720px max-width ---- */
.main {
  max-width: 1100px !important;
}

/* ---- Featured cards (horizontal) ---- */
.featured-list {
  display: flex;
  flex-direction: column;
  gap: 24px;
  margin-top: 8px;
}

.featured-card {
  display: flex;
  gap: 28px;
  align-items: flex-start;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  background: #ffffff;
  overflow: hidden;
  transition: box-shadow 0.2s ease;
}

.featured-card:hover {
  box-shadow: 0 6px 24px rgba(0,0,0,0.09);
}

.featured-img {
  flex-shrink: 0;
  width: 300px;
  height: 220px;
  object-fit: contain;
  background: #f9fafb;
  padding: 12px;
  align-self: stretch;
}

.featured-body {
  flex: 1;
  padding: 22px 24px 22px 0;
  min-width: 0;
}

.featured-title {
  font-size: 1.05em;
  font-weight: 700;
  color: #111827;
  margin: 0 0 10px 0;
  line-height: 1.4;
}

.featured-desc {
  font-size: 0.91em;
  color: #374151;
  line-height: 1.7;
  margin: 0 0 14px 0;
}

/* ---- Section divider ---- */
.section-label {
  font-size: 0.72em;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #9ca3af;
  margin: 44px 0 18px 0;
  padding-bottom: 8px;
  border-bottom: 1px solid #f3f4f6;
}

/* ---- Compact grid (older work) ---- */
.compact-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}

.compact-card {
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  background: #fafafa;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  transition: box-shadow 0.15s ease;
}

.compact-card:hover {
  box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}

.compact-img {
  width: 100%;
  height: 130px;
  object-fit: contain;
  display: block;
  border-bottom: 1px solid #f3f4f6;
  background: #f9fafb;
  padding: 10px;
  box-sizing: border-box;
}

.compact-body {
  padding: 13px 14px 14px 14px;
  display: flex;
  flex-direction: column;
  flex: 1;
}

.compact-title {
  font-size: 0.87em;
  font-weight: 700;
  color: #1f2937;
  margin: 0 0 8px 0;
  line-height: 1.4;
  flex: 1;
}

/* ---- Tags ---- */
.tag-row {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-bottom: 13px;
}

.tag {
  font-size: 0.67em;
  font-weight: 600;
  padding: 2px 7px;
  border-radius: 20px;
  background: #f3f4f6;
  color: #4b5563;
}

/* ---- Badges ---- */
.badge {
  display: inline-block;
  font-size: 0.7em;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 4px;
  background: #dcfce7;
  color: #15803d;
  margin-bottom: 8px;
}

/* ---- Buttons ---- */
.btn-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.btn {
  font-size: 0.82em;
  font-weight: 600;
  padding: 7px 15px;
  border-radius: 7px;
  text-decoration: none !important;
  display: inline-block;
  transition: background 0.15s ease, box-shadow 0.15s ease;
}

.btn-github {
  background: #2563eb !important;
  color: #ffffff !important;
  border: 1px solid #1d4ed8;
}

.btn-github:hover {
  background: #1d4ed8 !important;
  color: #ffffff !important;
  box-shadow: 0 2px 8px rgba(37,99,235,0.3);
}

.btn-demo {
  background: #f3f4f6 !important;
  color: #374151 !important;
  border: 1px solid #e5e7eb;
}

.btn-demo:hover {
  background: #e5e7eb !important;
  color: #111827 !important;
}

.btn-sm {
  font-size: 0.78em;
  padding: 5px 12px;
}

/* ---- Responsive ---- */
@media (max-width: 780px) {
  .featured-card { flex-direction: column; }
  .featured-img { width: 100%; height: 200px; padding: 16px; }
  .featured-body { padding: 16px; }
  .compact-grid { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 480px) {
  .compact-grid { grid-template-columns: 1fr; }
}
</style>

<div class="featured-list">

<div class="featured-card">
  <img src="/img/deep-research-agent/architecture.png" alt="Evidence-Driven Deep Research Agent architecture" class="featured-img">
  <div class="featured-body">
    <h3 class="featured-title">Evidence-Driven Deep Research Agent</h3>
    <p class="featured-desc">Research agent with an explicit evidence state, step-level reward signals, and a guided planner that uses those signals to decide what to search next. Runs an adversarial decomposer to generate counter-evidence sub-questions, a multi-stage retrieval pipeline with claim extraction and conflict detection, and a principled stopping criterion. Ablation across two queries: same compute envelope as baseline, full sub-question coverage vs. partial.</p>
    <div class="tag-row">
      <span class="tag">Agentic AI</span>
      <span class="tag">Process Rewards</span>
      <span class="tag">Python</span>
      <span class="tag">Tavily</span>
    </div>
    <div class="btn-row">
      <a href="https://github.com/rajathpatel23/deep-research-agent" class="btn btn-github">GitHub</a>
      <a href="/posts/deep-research-agent/" class="btn btn-demo">Blog Post</a>
    </div>
  </div>
</div>

<div class="featured-card">
  <img src="/img/project_images/labpilot_demo.jpg" alt="LabPilot demo" class="featured-img">
  <div class="featured-body">
    <h3 class="featured-title">LabPilot — AI Copilot for R&D Experiment Optimization</h3>
    <p class="featured-desc">Decision-support loop for R&D labs: surrogate model trained on historical experiment data recommends the next best experiment, with uncertainty quantification and adaptive bandit policies (UCB, LinUCB, greedy). An LLM reasoning layer explains each recommendation; a literature search layer adds source-backed justification. Full session loop — submit observed result, model adapts and recommends next step.</p>
    <div class="tag-row">
      <span class="tag">Bandit Policies</span>
      <span class="tag">Surrogate Modeling</span>
      <span class="tag">FastAPI</span>
      <span class="tag">React</span>
      <span class="tag">Nebius LLM</span>
    </div>
    <div class="btn-row">
      <a href="https://github.com/rajathpatel23/LabPilot" class="btn btn-github">GitHub</a>
      <a href="https://www.youtube.com/watch?v=7IyL28gGeqM" class="btn btn-demo">Demo Video</a>
    </div>
  </div>
</div>

<div class="featured-card">
  <img src="/img/project_images/emergency-agent.svg" alt="Emergency Guidance Agent FSM" class="featured-img" style="background:#f9fafb;">
  <div class="featured-body">
    <h3 class="featured-title">Emergency Guidance Agent — CPR Copilot</h3>
    <p class="featured-desc">Voice-and-video CPR coaching assistant using browser camera and microphone, built on Google Gemini Live. The core design decision: a strict finite state machine controls step transitions across six states (intake → escalation → see_patient → start_compressions → continue_cpr → complete). The model provides real-time guidance within each state; the application enforces the safety sequence. The model cannot advance or skip steps.</p>
    <div class="tag-row">
      <span class="tag">Multimodal</span>
      <span class="tag">Gemini Live</span>
      <span class="tag">FSM Safety Design</span>
      <span class="tag">TypeScript</span>
      <span class="tag">Pipecat</span>
    </div>
    <div class="btn-row">
      <a href="https://github.com/rajathpatel23/emergency-guidance-agent" class="btn btn-github">GitHub</a>
      <a href="https://youtu.be/DzXzaUUJiMg" class="btn btn-demo">Demo Video</a>
    </div>
  </div>
</div>

</div>

<div class="section-label">Earlier Work</div>

<div class="compact-grid">

<div class="compact-card">
  <img src="/img/project_images/introduction_slide.png" alt="Knowledge Graph Embedding" class="compact-img">
  <div class="compact-body">
    <h3 class="compact-title">Joint Knowledge Graph Embedding, Entity Typing, and Language Modeling</h3>
    <div class="tag-row">
      <span class="tag">NLP</span>
      <span class="tag">Knowledge Graphs</span>
      <span class="tag">Representation Learning</span>
    </div>
    <div class="btn-row">
      <a href="https://github.com/rajathpatel23/joint-kge-fnet-lm" class="btn btn-github btn-sm">GitHub</a>
    </div>
  </div>
</div>

<div class="compact-card">
  <img src="/img/project_images/CausalEventClassification.png" alt="Causal Event Classification" class="compact-img">
  <div class="compact-body">
    <div class="badge">2nd Place — F1: 84.36</div>
    <h3 class="compact-title">Supervised Contrastive Learning for Causal Event Classification</h3>
    <div class="tag-row">
      <span class="tag">Contrastive Learning</span>
      <span class="tag">NLP</span>
      <span class="tag">Shared Task</span>
    </div>
    <div class="btn-row">
      <a href="https://github.com/rajathpatel23/causal-events" class="btn btn-github btn-sm">GitHub</a>
    </div>
  </div>
</div>

<div class="compact-card">
  <img src="/img/project_images/Know_rep.png" alt="Image Recognition with Knowledge Graph" class="compact-img">
  <div class="compact-body">
    <h3 class="compact-title">Image Recognition with Knowledge Graph Embedding</h3>
    <div class="tag-row">
      <span class="tag">Vision</span>
      <span class="tag">Knowledge Graphs</span>
      <span class="tag">Joint Learning</span>
    </div>
    <div class="btn-row">
      <a href="https://github.com/rajathpatel23/object_recog_KGE" class="btn btn-github btn-sm">GitHub</a>
    </div>
  </div>
</div>

<div class="compact-card">
  <img src="/img/project_images/HMM.jpg" alt="Hidden Markov Models" class="compact-img">
  <div class="compact-body">
    <h3 class="compact-title">Hidden Markov Models — Forward-Backward and Viterbi</h3>
    <div class="tag-row">
      <span class="tag">Probabilistic Models</span>
      <span class="tag">NLP</span>
      <span class="tag">Python</span>
    </div>
    <div class="btn-row">
      <a href="https://github.com/rajathpatel23/hmm-implementation" class="btn btn-github btn-sm">GitHub</a>
    </div>
  </div>
</div>

<div class="compact-card">
  <img src="/img/project_images/mixed_norms_new.png" alt="FISTA ADMM" class="compact-img">
  <div class="compact-body">
    <h3 class="compact-title">FISTA, ADMM, and Mixed Norms</h3>
    <div class="tag-row">
      <span class="tag">Optimization</span>
      <span class="tag">Python</span>
    </div>
    <div class="btn-row">
      <a href="https://github.com/shimingyoung/mtl" class="btn btn-github btn-sm">GitHub</a>
    </div>
  </div>
</div>

<div class="compact-card">
  <div class="compact-body" style="padding-top: 18px;">
    <h3 class="compact-title">Streaming Tweets with Kafka</h3>
    <div class="tag-row">
      <span class="tag">Kafka</span>
      <span class="tag">Streaming</span>
      <span class="tag">Python</span>
    </div>
    <div class="btn-row">
      <a href="https://github.com/rajathpatel23/kafka-stream-tweets" class="btn btn-github btn-sm">GitHub</a>
    </div>
  </div>
</div>

</div>
