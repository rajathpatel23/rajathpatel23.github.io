---
title: "Projects"
date: 2025-01-27
draft: false
showToc: false
TocOpen: false
---

<style>
.tiles-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  margin-top: 24px;
}

/* ---- tile card ---- */
.tile {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  background: #ffffff;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  transition: box-shadow 0.2s ease, transform 0.2s ease;
}

.tile:hover {
  box-shadow: 0 6px 20px rgba(0,0,0,0.09);
  transform: translateY(-2px);
}

/* ---- image ---- */
.tile-img-wrap {
  position: relative;
}

.tile-img {
  width: 100%;
  height: 160px;
  object-fit: cover;
  display: block;
  border-bottom: 1px solid #f3f4f6;
  background: #f9fafb;
}

.tile-badge {
  position: absolute;
  top: 10px;
  left: 10px;
  font-size: 0.7em;
  font-weight: 700;
  padding: 3px 8px;
  border-radius: 4px;
  background: #dcfce7;
  color: #15803d;
  letter-spacing: 0.02em;
}

/* ---- body ---- */
.tile-body {
  padding: 16px 18px 18px 18px;
  display: flex;
  flex-direction: column;
  flex: 1;
}

.tile-title {
  font-size: 0.97em;
  font-weight: 700;
  color: #111827;
  margin: 0 0 10px 0;
  line-height: 1.4;
}

/* ---- tags ---- */
.tag-row {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-bottom: 12px;
}

.tag {
  font-size: 0.68em;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 20px;
  background: #f3f4f6;
  color: #4b5563;
  letter-spacing: 0.02em;
}

/* ---- collapsible description ---- */
.tile-details {
  margin-bottom: 14px;
  flex: 1;
}

.tile-details summary {
  cursor: pointer;
  font-size: 0.82em;
  font-weight: 600;
  color: #6b7280;
  list-style: none;
  display: flex;
  align-items: center;
  gap: 5px;
  user-select: none;
  padding: 4px 0;
}

.tile-details summary::-webkit-details-marker { display: none; }

.tile-details summary::after {
  content: "▾";
  font-size: 0.9em;
  transition: transform 0.2s ease;
  display: inline-block;
}

.tile-details[open] summary::after {
  transform: rotate(180deg);
}

.tile-details summary:hover {
  color: #374151;
}

.tile-desc {
  font-size: 0.87em;
  color: #4b5563;
  line-height: 1.65;
  margin: 8px 0 0 0;
}

/* ---- buttons ---- */
.tile-links {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: auto;
}

.btn {
  font-size: 0.82em;
  font-weight: 600;
  padding: 7px 15px;
  border-radius: 7px;
  text-decoration: none !important;
  transition: background 0.2s ease, box-shadow 0.2s ease;
  display: inline-block;
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

.btn-secondary {
  background: #f3f4f6 !important;
  color: #374151 !important;
  border: 1px solid #e5e7eb;
}

.btn-secondary:hover {
  background: #e5e7eb !important;
  color: #111827 !important;
}

/* ---- responsive ---- */
@media (max-width: 900px) {
  .tiles-grid { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 560px) {
  .tiles-grid { grid-template-columns: 1fr; }
}
</style>

<div class="tiles-grid">

<div class="tile">
  <div class="tile-img-wrap">
    <img src="/img/deep-research-agent/architecture.png" alt="Evidence-Driven Deep Research Agent" class="tile-img">
  </div>
  <div class="tile-body">
    <h3 class="tile-title">Evidence-Driven Deep Research Agent</h3>
    <div class="tag-row">
      <span class="tag">Agentic AI</span>
      <span class="tag">Process Rewards</span>
      <span class="tag">Python</span>
      <span class="tag">Tavily</span>
    </div>
    <details class="tile-details">
      <summary>About</summary>
      <p class="tile-desc">Research agent with an explicit evidence state, step-level reward signals, and a guided planner that uses those signals to decide what to search next. Includes an ablation comparing baseline vs guided planner across two queries — same compute envelope, better coverage.</p>
    </details>
    <div class="tile-links">
      <a href="https://github.com/rajathpatel23/deep-research-agent" class="btn btn-github">GitHub</a>
      <a href="/posts/deep-research-agent/" class="btn btn-secondary">Blog Post</a>
    </div>
  </div>
</div>

<div class="tile">
  <div class="tile-img-wrap">
    <img src="/img/project_images/labpilot_demo.jpg" alt="LabPilot" class="tile-img">
  </div>
  <div class="tile-body">
    <h3 class="tile-title">LabPilot — AI Copilot for R&D Experiment Optimization</h3>
    <div class="tag-row">
      <span class="tag">Bandit Policies</span>
      <span class="tag">scikit-learn</span>
      <span class="tag">FastAPI</span>
      <span class="tag">React</span>
      <span class="tag">Nebius LLM</span>
    </div>
    <details class="tile-details">
      <summary>About</summary>
      <p class="tile-desc">Decision-support loop for R&D labs: surrogate model trained on historical data recommends the next best experiment, with uncertainty quantification, adaptive bandit policies (UCB, LinUCB), and LLM-powered explanations backed by literature search.</p>
    </details>
    <div class="tile-links">
      <a href="https://github.com/rajathpatel23/LabPilot" class="btn btn-github">GitHub</a>
      <a href="https://www.youtube.com/watch?v=7IyL28gGeqM" class="btn btn-secondary">Demo Video</a>
    </div>
  </div>
</div>

<div class="tile">
  <div class="tile-img-wrap">
    <img src="/img/project_images/emergency-agent.svg" alt="Emergency Guidance Agent" class="tile-img" style="background:#f9fafb;">
  </div>
  <div class="tile-body">
    <h3 class="tile-title">Emergency Guidance Agent — CPR Copilot</h3>
    <div class="tag-row">
      <span class="tag">Multimodal</span>
      <span class="tag">Gemini Live</span>
      <span class="tag">FSM Safety</span>
      <span class="tag">TypeScript</span>
      <span class="tag">Pipecat</span>
    </div>
    <details class="tile-details">
      <summary>About</summary>
      <p class="tile-desc">Voice-and-video CPR coaching assistant using browser camera and mic. Built on Google Gemini Live. Key design: a strict FSM controls step transitions — the model guides within each state, but the app enforces the safety sequence. The model cannot advance or skip steps.</p>
    </details>
    <div class="tile-links">
      <a href="https://github.com/rajathpatel23/emergency-guidance-agent" class="btn btn-github">GitHub</a>
      <a href="https://youtu.be/DzXzaUUJiMg" class="btn btn-secondary">Demo Video</a>
    </div>
  </div>
</div>

<div class="tile">
  <div class="tile-img-wrap">
    <img src="/img/project_images/introduction_slide.png" alt="Knowledge Graph Embedding" class="tile-img">
  </div>
  <div class="tile-body">
    <h3 class="tile-title">Joint Knowledge Graph Embedding, Entity Typing, and Language Modeling</h3>
    <div class="tag-row">
      <span class="tag">NLP</span>
      <span class="tag">Knowledge Graphs</span>
      <span class="tag">Representation Learning</span>
    </div>
    <details class="tile-details">
      <summary>About</summary>
      <p class="tile-desc">Shows the complementary nature of neural KGE, fine-grain entity type prediction, and language modeling. A language model-inspired KGE approach yields improved embeddings and entity type representations simultaneously.</p>
    </details>
    <div class="tile-links">
      <a href="https://github.com/rajathpatel23/joint-kge-fnet-lm" class="btn btn-github">GitHub</a>
    </div>
  </div>
</div>

<div class="tile">
  <div class="tile-img-wrap">
    <img src="/img/project_images/CausalEventClassification.png" alt="Causal Event Classification" class="tile-img">
    <span class="tile-badge">2nd Place — F1: 84.36</span>
  </div>
  <div class="tile-body">
    <h3 class="tile-title">Supervised Contrastive Learning for Causal Event Classification</h3>
    <div class="tag-row">
      <span class="tag">Contrastive Learning</span>
      <span class="tag">NLP</span>
      <span class="tag">Shared Task</span>
    </div>
    <details class="tile-details">
      <summary>About</summary>
      <p class="tile-desc">Causal News Corpus shared task 2023. Pre-trains with Supervised Contrastive (SuperCon) learning, then fine-tunes for causal event classification.</p>
    </details>
    <div class="tile-links">
      <a href="https://github.com/rajathpatel23/causal-events" class="btn btn-github">GitHub</a>
    </div>
  </div>
</div>

<div class="tile">
  <div class="tile-img-wrap">
    <img src="/img/project_images/Know_rep.png" alt="Image Recognition with Knowledge Graph" class="tile-img">
  </div>
  <div class="tile-body">
    <h3 class="tile-title">Image Recognition with Knowledge Graph Embedding</h3>
    <div class="tag-row">
      <span class="tag">Vision</span>
      <span class="tag">Knowledge Graphs</span>
      <span class="tag">Joint Learning</span>
    </div>
    <details class="tile-details">
      <summary>About</summary>
      <p class="tile-desc">Joint learning model for images and image-captioned entity attribute representations to learn semantic relationships from knowledge graph embeddings.</p>
    </details>
    <div class="tile-links">
      <a href="https://github.com/rajathpatel23/object_recog_KGE" class="btn btn-github">GitHub</a>
    </div>
  </div>
</div>

<div class="tile">
  <div class="tile-img-wrap">
    <img src="/img/project_images/HMM.jpg" alt="Hidden Markov Models" class="tile-img">
  </div>
  <div class="tile-body">
    <h3 class="tile-title">Hidden Markov Models — Forward-Backward and Viterbi</h3>
    <div class="tag-row">
      <span class="tag">Probabilistic Models</span>
      <span class="tag">NLP</span>
      <span class="tag">Python</span>
    </div>
    <details class="tile-details">
      <summary>About</summary>
      <p class="tile-desc">Python implementation of HMM algorithms (forward-backward, Viterbi) applied to POS tagging.</p>
    </details>
    <div class="tile-links">
      <a href="https://github.com/rajathpatel23/hmm-implementation" class="btn btn-github">GitHub</a>
    </div>
  </div>
</div>

<div class="tile">
  <div class="tile-img-wrap">
    <img src="/img/project_images/mixed_norms_new.png" alt="FISTA ADMM" class="tile-img">
  </div>
  <div class="tile-body">
    <h3 class="tile-title">FISTA, ADMM, and Mixed Norms</h3>
    <div class="tag-row">
      <span class="tag">Optimization</span>
      <span class="tag">Python</span>
    </div>
    <details class="tile-details">
      <summary>About</summary>
      <p class="tile-desc">Python implementation of convex optimization algorithms: FISTA, ADMM, and mixed norm regularization.</p>
    </details>
    <div class="tile-links">
      <a href="https://github.com/shimingyoung/mtl" class="btn btn-github">GitHub</a>
    </div>
  </div>
</div>

<div class="tile">
  <div class="tile-body" style="padding-top: 24px;">
    <h3 class="tile-title">Streaming Tweets with Kafka</h3>
    <div class="tag-row">
      <span class="tag">Kafka</span>
      <span class="tag">Streaming</span>
      <span class="tag">Python</span>
    </div>
    <details class="tile-details">
      <summary>About</summary>
      <p class="tile-desc">Real-time tweet extraction and streaming pipeline using Kafka, designed for downstream ML model integration.</p>
    </details>
    <div class="tile-links">
      <a href="https://github.com/rajathpatel23/kafka-stream-tweets" class="btn btn-github">GitHub</a>
    </div>
  </div>
</div>

</div>
