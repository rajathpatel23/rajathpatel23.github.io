---
title: "Projects"
date: 2025-01-27
draft: false
showToc: false
TocOpen: false
---

<style>
/* ---- layout ---- */
.projects-section-label {
  font-size: 0.75em;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #6b7280;
  margin: 48px 0 20px 0;
  padding-bottom: 8px;
  border-bottom: 1px solid #e5e7eb;
}

.projects-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 12px;
}

.projects-grid-single {
  display: grid;
  grid-template-columns: 1fr;
  gap: 24px;
  margin-bottom: 12px;
}

/* ---- cards ---- */
.project-card {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  background: #ffffff;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  transition: box-shadow 0.2s ease, transform 0.2s ease;
}

.project-card:hover {
  box-shadow: 0 6px 20px rgba(0,0,0,0.10);
  transform: translateY(-2px);
}

.project-card-image {
  width: 100%;
  height: 180px;
  object-fit: cover;
  display: block;
  border-bottom: 1px solid #f3f4f6;
}

.project-card-body {
  padding: 20px 22px 22px 22px;
  flex: 1;
  display: flex;
  flex-direction: column;
}

.project-card-title {
  font-size: 1.05em;
  font-weight: 700;
  color: #111827;
  margin: 0 0 10px 0;
  line-height: 1.4;
}

.project-card-desc {
  font-size: 0.92em;
  color: #374151;
  line-height: 1.65;
  margin: 0 0 14px 0;
  flex: 1;
}

/* ---- tags ---- */
.tag-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 16px;
}

.tag {
  font-size: 0.72em;
  font-weight: 600;
  padding: 3px 9px;
  border-radius: 20px;
  background: #f3f4f6;
  color: #374151;
  letter-spacing: 0.02em;
}

/* ---- links ---- */
.card-links {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.card-link {
  font-size: 0.85em;
  font-weight: 600;
  padding: 7px 16px;
  border-radius: 7px;
  text-decoration: none;
  transition: background 0.2s ease;
}

.card-link-primary {
  background: #111827;
  color: #ffffff;
}

.card-link-primary:hover {
  background: #1f2937;
  color: #ffffff;
  text-decoration: none;
}

.card-link-secondary {
  background: #f3f4f6;
  color: #374151;
  border: 1px solid #e5e7eb;
}

.card-link-secondary:hover {
  background: #e5e7eb;
  color: #111827;
  text-decoration: none;
}

/* ---- older work compact cards ---- */
.project-card-compact {
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  background: #fafafa;
  display: flex;
  gap: 20px;
  padding: 18px 20px;
  align-items: flex-start;
  transition: box-shadow 0.2s ease;
}

.project-card-compact:hover {
  box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}

.compact-img {
  flex-shrink: 0;
  width: 100px;
  height: 72px;
  object-fit: cover;
  border-radius: 7px;
  border: 1px solid #e5e7eb;
}

.compact-body {
  flex: 1;
  min-width: 0;
}

.compact-title {
  font-size: 0.95em;
  font-weight: 700;
  color: #111827;
  margin: 0 0 6px 0;
  line-height: 1.4;
}

.compact-desc {
  font-size: 0.87em;
  color: #4b5563;
  line-height: 1.6;
  margin: 0 0 10px 0;
}

.compact-link {
  font-size: 0.82em;
  font-weight: 600;
  color: #2563eb;
  text-decoration: none;
}

.compact-link:hover {
  text-decoration: underline;
}

.achievement-tag {
  display: inline-block;
  font-size: 0.72em;
  font-weight: 700;
  padding: 2px 8px;
  background: #dcfce7;
  color: #15803d;
  border-radius: 4px;
  margin-bottom: 6px;
  letter-spacing: 0.02em;
}

/* ---- responsive ---- */
@media (max-width: 680px) {
  .projects-grid {
    grid-template-columns: 1fr;
  }
  .project-card-compact {
    flex-direction: column;
  }
  .compact-img {
    width: 100%;
    height: 140px;
  }
}
</style>

<div class="projects-section-label">Recent Projects</div>

<div class="projects-grid">

<div class="project-card">
  <img src="/img/deep-research-agent/architecture.png" alt="Evidence-Driven Deep Research Agent architecture" class="project-card-image">
  <div class="project-card-body">
    <h3 class="project-card-title">Evidence-Driven Deep Research Agent</h3>
    <p class="project-card-desc">Research agent with an explicit evidence state, step-level reward signals, and a planner that uses those signals to decide what to search next. Includes an ablation comparing baseline vs guided planner modes across two queries.</p>
    <div class="tag-row">
      <span class="tag">Agentic AI</span>
      <span class="tag">Process Rewards</span>
      <span class="tag">Python</span>
      <span class="tag">Tavily</span>
    </div>
    <div class="card-links">
      <a href="https://github.com/rajathpatel23/deep-research-agent" class="card-link card-link-primary">GitHub</a>
      <a href="/posts/deep-research-agent/" class="card-link card-link-secondary">Blog Post</a>
    </div>
  </div>
</div>

<div class="project-card">
  <img src="/img/project_images/labpilot_demo.jpg" alt="LabPilot demo" class="project-card-image">
  <div class="project-card-body">
    <h3 class="project-card-title">LabPilot — AI Copilot for R&D Experiment Optimization</h3>
    <p class="project-card-desc">Decision-support loop for R&D labs: surrogate model trained on historical data recommends the next best experiment, with uncertainty quantification, adaptive bandit policies (UCB, LinUCB), and LLM-powered explanations backed by literature search.</p>
    <div class="tag-row">
      <span class="tag">Bandit Policies</span>
      <span class="tag">scikit-learn</span>
      <span class="tag">FastAPI</span>
      <span class="tag">React</span>
      <span class="tag">Nebius LLM</span>
    </div>
    <div class="card-links">
      <a href="https://github.com/rajathpatel23/LabPilot" class="card-link card-link-primary">GitHub</a>
      <a href="https://www.youtube.com/watch?v=7IyL28gGeqM" class="card-link card-link-secondary">Demo Video</a>
    </div>
  </div>
</div>

</div>

<div class="projects-grid-single">

<div class="project-card">
  <img src="/img/project_images/emergency-agent.svg" alt="Emergency Guidance Agent FSM architecture" class="project-card-image" style="height:200px; background:#f9fafb;">
  <div class="project-card-body">
    <h3 class="project-card-title">Emergency Guidance Agent — CPR Copilot</h3>
    <p class="project-card-desc">Voice-and-video CPR coaching assistant that walks lay rescuers through adult CPR steps in real time using browser camera and microphone. Built on Google Gemini Live (audio + video). The key design decision: a strict finite state machine controls step transitions — the model provides guidance within each state, but the app enforces the safety sequence. The model never gets to decide when to advance steps.</p>
    <div class="tag-row">
      <span class="tag">Multimodal</span>
      <span class="tag">Gemini Live</span>
      <span class="tag">FSM Safety Design</span>
      <span class="tag">TypeScript</span>
      <span class="tag">Pipecat</span>
      <span class="tag">FastAPI</span>
    </div>
    <div class="card-links">
      <a href="https://github.com/rajathpatel23/emergency-guidance-agent" class="card-link card-link-primary">GitHub</a>
    </div>
  </div>
</div>

</div>

<div class="projects-section-label">Earlier Work</div>

<div style="display: flex; flex-direction: column; gap: 16px;">

<div class="project-card-compact">
  <img src="/img/project_images/introduction_slide.png" alt="Knowledge Graph Embedding" class="compact-img">
  <div class="compact-body">
    <h3 class="compact-title">Joint Knowledge Graph Embedding, Fine Grain Entity Typing, and Language Modeling</h3>
    <p class="compact-desc">Shows the complementary nature of neural KGE, fine-grain entity type prediction, and neural language modeling. A language model-inspired KGE approach yields improved embeddings and entity type representations simultaneously.</p>
    <a href="https://github.com/rajathpatel23/joint-kge-fnet-lm" class="compact-link">View on GitHub</a>
  </div>
</div>

<div class="project-card-compact">
  <img src="/img/project_images/CausalEventClassification.png" alt="Causal Event Classification" class="compact-img">
  <div class="compact-body">
    <div class="achievement-tag">2nd Place — F1: 84.36</div>
    <h3 class="compact-title">Supervised Contrastive Learning for Causal Event Classification</h3>
    <p class="compact-desc">Causal News Corpus shared task 2023. Pre-trains with Supervised Contrastive (SuperCon) learning, then fine-tunes for causal event classification.</p>
    <a href="https://github.com/rajathpatel23/causal-events" class="compact-link">View on GitHub</a>
  </div>
</div>

<div class="project-card-compact">
  <img src="/img/project_images/Know_rep.png" alt="Image Recognition with Knowledge Graph" class="compact-img">
  <div class="compact-body">
    <h3 class="compact-title">Image Recognition with Knowledge Graph Embedding</h3>
    <p class="compact-desc">Joint learning model for images and image-captioned entity attribute representations to learn semantic relationships from knowledge graph embeddings.</p>
    <a href="https://github.com/rajathpatel23/object_recog_KGE" class="compact-link">View on GitHub</a>
  </div>
</div>

<div class="project-card-compact">
  <img src="/img/project_images/HMM.jpg" alt="Hidden Markov Models" class="compact-img">
  <div class="compact-body">
    <h3 class="compact-title">Hidden Markov Models — Forward-Backward and Viterbi</h3>
    <p class="compact-desc">Python implementation of HMM algorithms (forward-backward, Viterbi) applied to POS tagging.</p>
    <a href="https://github.com/rajathpatel23/hmm-implementation" class="compact-link">View on GitHub</a>
  </div>
</div>

<div class="project-card-compact">
  <img src="/img/project_images/mixed_norms_new.png" alt="FISTA ADMM" class="compact-img">
  <div class="compact-body">
    <h3 class="compact-title">FISTA, ADMM, and Mixed Norms</h3>
    <p class="compact-desc">Python implementation of optimization algorithms: FISTA, ADMM, and mixed norm regularization.</p>
    <a href="https://github.com/shimingyoung/mtl" class="compact-link">View on GitHub</a>
  </div>
</div>

<div class="project-card-compact">
  <div class="compact-body">
    <h3 class="compact-title">Streaming Tweets with Kafka</h3>
    <p class="compact-desc">Real-time tweet extraction and streaming pipeline using Kafka, designed for downstream ML model integration.</p>
    <a href="https://github.com/rajathpatel23/kafka-stream-tweets" class="compact-link">View on GitHub</a>
  </div>
</div>

</div>
