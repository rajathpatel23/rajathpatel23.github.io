---
title: "Projects"
date: 2025-01-27
draft: false
showToc: true
TocOpen: true
---

## 🚀 Projects

<style>
.project-card {
  display: flex;
  align-items: flex-start;
  gap: 30px;
  margin-bottom: 50px;
  padding: 25px;
  border: 2px solid #e5e7eb;
  border-radius: 16px;
  background: #ffffff;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.project-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}

.project-image {
  flex-shrink: 0;
  width: 280px;
  height: 200px;
  object-fit: cover;
  border-radius: 12px;
  box-shadow: 0 6px 16px rgba(0,0,0,0.2);
  border: 3px solid #f3f4f6;
  transition: transform 0.3s ease;
}

.project-image:hover {
  transform: scale(1.05);
}

.project-content {
  flex: 1;
  min-width: 0;
}

.project-title {
  margin: 0 0 15px 0;
  color: #1f2937;
  font-size: 1.4em;
  font-weight: 700;
  line-height: 1.3;
}

.project-description {
  margin-bottom: 20px;
  line-height: 1.7;
  color: #374151;
  font-size: 1.05em;
}

.project-link {
  display: inline-block;
  padding: 12px 24px;
  background: #2563eb;
  color: white;
  text-decoration: none;
  border-radius: 8px;
  font-weight: 600;
  font-size: 1.05em;
  transition: background 0.3s ease, transform 0.2s ease;
}

.project-link:hover {
  background: #1d4ed8;
  color: white;
  text-decoration: none;
  transform: translateY(-1px);
}

.achievement {
  background: #dcfce7;
  color: #166534;
  padding: 8px 16px;
  border-radius: 25px;
  font-size: 0.95em;
  font-weight: 700;
  display: inline-block;
  margin: 0 0 15px 0;
  border: 2px solid #86efac;
}

@media (max-width: 968px) {
  .project-card {
    flex-direction: column;
    text-align: center;
    padding: 20px;
  }
  
  .project-image {
    width: 100%;
    max-width: 400px;
    height: 250px;
    margin: 0 auto 20px auto;
  }
  
  .project-title {
    font-size: 1.25em;
  }
}

@media (max-width: 640px) {
  .project-image {
    height: 200px;
  }
}
</style>

<div class="project-card">
  <img src="/img/project_images/introduction_slide.png" alt="Knowledge Graph Embedding" class="project-image">
  <div class="project-content">
    <h3 class="project-title">Joint Knowledge Graph Embedding, Fine Grain Entity Type and Language Modeling</h3>
    <p class="project-description">
      We demonstrate the complementary natures of neural knowledge graph embedding, finegrain entity type prediction, and neural language modeling. We show that a language model-inspired knowledge graph embedding approach yields both improved knowledge graph embeddings and fine-grain entity type representations. Our work also shows that jointly modeling both structured knowledge tuples and language improves both.
    </p>
    <a href="https://github.com/rajathpatel23/joint-kge-fnet-lm" class="project-link">📚 View on GitHub</a>
  </div>
</div>

<div class="project-card">
  <img src="/img/project_images/CausalEventClassification.png" alt="Causal Event Classification" class="project-image">
  <div class="project-content">
    <h3 class="project-title">Understanding Causal Relationships: Supervised Contrastive Learning for Event Classification</h3>
    <div class="achievement">🏆 2nd Position - F1-Score: 84.36</div>
    <p class="project-description">
      Causal events play a crucial role in explaining the intricate relationships between the causes and effects of events. We propose a contrastive learning-based method for the Causal News Corpus - Event Causality Shared Task 2023, focusing on Subtask 1 centered on causal event classification. Our approach pre-trains the base model using Supervised Contrastive (SuperCon) learning, then fine-tunes for causal event classification.
    </p>
    <a href="https://github.com/rajathpatel23/causal-events" class="project-link">🔗 View on GitHub</a>
  </div>
</div>

<div class="project-card">
  <img src="/img/project_images/Know_rep.png" alt="Image Recognition with Knowledge Graph" class="project-image">
  <div class="project-content">
    <h3 class="project-title">Combining Image Recognition with Knowledge Graph Embedding for Learning Semantic Attribute of Images</h3>
    <p class="project-description">
      Linking entities in the knowledge graph has been an important problem. Learning of images in the open world using language model has attracted lots of interest over the year. Through this paper, we propose a joint learning model to learn images along with image captioned entity attribute representation to learn the semantic relationships from the knowledge graph embedding model. The target model premises to help us understand the semantic relationship between the attribute entities and implicitly provide a link prediction for these entities.
    </p>
    <a href="https://github.com/rajathpatel23/object_recog_KGE" class="project-link">🖼️ View on GitHub</a>
  </div>
</div>

<div class="project-card">
  <img src="/img/project_images/HMM.jpg" alt="Hidden Markov Models" class="project-image">
  <div class="project-content">
    <h3 class="project-title">Hidden Markov Models Implementation</h3>
    <p class="project-description">
      Python implementation of Hidden Markov Models (HMM). The implementation provides source code for forward-backward algorithm and viterbi algorithm for part of speech tagging problem using HMM. This project demonstrates the practical application of probabilistic models in natural language processing tasks.
    </p>
    <a href="https://github.com/rajathpatel23/hmm-implementation" class="project-link">⚡ View on GitHub</a>
  </div>
</div> 

<div class="project-card">
  <img src="/img/project_images/mixed_norms_new.png" alt="FISTA, ADMM implementation" class="project-image">
  <div class="project-content">
    <h3 class="project-title">FISTA, ADMM implementation</h3>
    <p class="project-description">
      The project provides python
      implementation of optimization algorithms like FISTA, ADMM and mixed norms.
    </p>
    <a href="hhttps://github.com/shimingyoung/mtl" class="project-link">⚡ View on GitHub</a>
  </div>
</div> 

<div class="project-card">
  <!-- <img src="img/project_images/Kafka_twitter_new.png" alt="Streaming Tweets with Kafka<" class="project-image"> -->
  <div class="project-content">
    <h3 class="project-title">Streaming Tweets with Kafka</h3>
    <p class="project-description">
      This project gives understanding of the streaming capability with Kafka for extracting tweets from twitter.
      The Kafka streaming service can be used to build pipelines to analyzing the tweets in real time with a ML models.
    </p>
    <a href="https://github.com/rajathpatel23/kafka-stream-tweets" class="project-link">⚡ View on GitHub</a>
  </div>
</div> 