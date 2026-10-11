---
title: "Pretrain, SFT, then the DPO run that held"
date: 2026-09-27
draft: false
showToc: true
TocOpen: false
tags: ["post-training", "SFT", "DPO", "language-models"]
categories: ["Technical"]
description: "A 15M-parameter model pretrained on tiny-shakespeare, fine-tuned to stop after one line, then aligned with DPO. The same learning rate that worked for SFT broke generation. A tenfold cut fixed it."
---

I pretrained TinyGPT on tiny-shakespeare, fine-tuned it to say one sentence and stop, then used DPO to prefer a fuller line. TinyGPT has 12 layers, 8 heads, 128-dimensional embeddings, the GPT-2 byte-pair vocabulary, and about 15 million parameters. The pretrained checkpoint is `out_bpe/tinygpt_shakespeare.pt`. DPO at the fine-tuning learning rate raised the metrics and broke the generated text. The run kept here cuts that rate by ten. Every comparison uses `ROMEO:`, `MENENIUS:`, `JULIET:`, and `KING RICHARD III:`.

## The runs

DPO was started four times from the fine-tuned checkpoint. I kept the fourth.

<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
  mermaid.initialize({
    startOnLoad: true,
    theme: "base",
    securityLevel: "loose",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    flowchart: { curve: "basis", padding: 18, nodeSpacing: 36, rankSpacing: 52, htmlLabels: true },
    themeVariables: {
      fontSize: "15px",
      background: "transparent",
      primaryColor: "#f7f7f7",
      primaryTextColor: "#1e1e1e",
      primaryBorderColor: "#d0d0d0",
      lineColor: "#6c6c6c",
      secondaryColor: "#f7f7f7",
      tertiaryColor: "#f7f7f7"
    }
  });
</script>

<pre class="mermaid">
flowchart LR
  pre["Pretrain\nnext token"]
  sft["Fine-tune\none sentence, then stop"]
  dpo["DPO\nprefer one reply"]
  pre --> sft --> dpo
</pre>

## Pretraining

Pretraining predicts the next byte-pair token. For a window $x_1, \ldots, x_T$, the loss is the average cross-entropy of the next token:

$$
\mathcal{L}\_{\mathrm{pretrain}} = -\frac{1}{T}\sum\_{t=1}^{T} \log p\_\theta(x\_{t+1} \mid x\_{\le t})
$$

Tiny-shakespeare is one long file. It has no document boundary, so the end-of-text token is never a target, and that row of the embedding stays random. A loss in subword units is a different number from a character-level loss on the same file. It also says nothing about whether a line is finished.

Two thousand steps, batch size 8, learning rate `3e-4`. Training loss fell from 10.83 to 3.84. Validation loss fell from 10.82 to 4.78. Validation perplexity ended near 120. The curves separate after a few hundred steps.

![Loss and perplexity from the pretraining script](/img/posts/tiny-shakespeare/bpe_pretrain_curves.png)

The W&B run is [tinygpt-bpe-v1](https://wandb.ai/rajat-patel/nanogpt-shakespeare/runs/pmt5c5vx).

Asked for Romeo, it holds the shape for a clause and then walks into other characters:

<div class="folio">
  <div class="cue asked">
    <div class="who">ROMEO:</div>
    <div class="said">And I will not a man.</div>
  </div>
  <div class="cue">
    <div class="who">LADYea,</div>
    <div class="said">I do you.</div>
  </div>
  <div class="cue">
    <div class="who">ROMEO:</div>
    <div class="said">I'll not my lord.</div>
  </div>
  <div class="cue">
    <div class="who">WARWICK:</div>
    <div class="said">Why woe no more than thou wast thou art<br>The word is a prince,<br>I'll find your grace, my lord; and let him.</div>
  </div>
  <div class="cue">
    <div class="who">Second Senator:</div>
    <div class="said">Rome are it;</div>
  </div>
</div>

Pretraining got us a model of the file. A clause can sound right, and then the next tokens are whoever speaks next, because that is what the file does. More steps on the same file would only make that continuation smoother. The next step is fine-tuning: take a speaker, say one sentence, and stop.

The training script is [`train.py`](https://github.com/rajathpatel23/post-training-roadmap/blob/main/scratch_nanogpt/train.py), run with [`gpt_small_bpe.yaml`](https://github.com/rajathpatel23/post-training-roadmap/blob/main/scratch_nanogpt/configs/train/gpt_small_bpe.yaml).

## Fine-tuning

I wanted a new behavior: given a speaker, say one sentence and stop. The raw play treats that stop as a mistake, because the likely next tokens are the rest of the scene. Training on the raw file again would have practiced the drift.

### Speaker, then one sentence

Blank lines separate speeches. A prompt is the first line of a speech. The target is the first sentence after it, with the end-of-text token appended. Pretraining never asked for that token, and sampling needs a stop.

| | Rule |
|---|---|
| Prompt | ends in `:`, at most 30 characters, no `.`, `?`, or `!` before the colon |
| Target | the speech joined across wraps, then the first sentence of 8 to 150 characters ending in `.`, `?`, or `!` |

| | Second Citizen |
|---|---|
| Prompt | Second Citizen: |
| Target | What he cannot help in his nature, you account a vice in him. |
| Left out | You must in no way say he is covetous. |

The break after "account a" is the line wrap. The cut is the period. "Cousin, farewell:" ends in a colon and is still dialogue, so it is not a prompt.

That filter made 6,344 pairs. Shuffled with seed 1337, 5,710 went to training and 634 to eval. The filter is [`sft_data.py`](https://github.com/rajathpatel23/post-training-roadmap/blob/main/scratch_nanogpt/sft_data.py). The pairs are in [`sft.csv`](https://github.com/rajathpatel23/post-training-roadmap/blob/main/scratch_nanogpt/datasets/sft.csv).

### Training

The loss is the same cross-entropy, on fewer positions. Prompt tokens are labeled `-100` and drop out. The reply, including the end-of-text token, stays in. The mean is over those tokens, so a longer line counts more than a short one:

$$
\mathcal{L}\_{\mathrm{SFT}} = -\frac{1}{|y|}\sum\_{t=1}^{|y|} \log p\_\theta(y\_t \mid \text{speaker},\, y\_{<t})
$$

Training was 3 epochs, 1,000 steps, batch size 16, learning rate `5e-5`, about 83 seconds. The script is [`train_sft.py`](https://github.com/rajathpatel23/post-training-roadmap/blob/main/scratch_nanogpt/train_sft.py) with [`sft_bpe.yaml`](https://github.com/rajathpatel23/post-training-roadmap/blob/main/scratch_nanogpt/configs/train/sft_bpe.yaml). The W&B run for these 1,000 steps is [tinygpt-bpe-sft-v1](https://wandb.ai/rajat-patel/nanogpt-shakespeare/runs/pz05y6tq).

### Results

![Train and validation cross-entropy from the fine-tuning script](/img/posts/tiny-shakespeare/sft_loss.png)

Validation loss falls from 4.91 at step 100 to 4.35 at step 1,000, then flattens. The train line is a single batch at each logged step, so it jumps around. The same prompts stopped:

| Prompt | After supervised fine-tuning |
|---|---|
| ROMEO: | I will not be gone, and see the king. |
| MENENIUS: | You hear us! |
| JULIET: | What, that knows not to do not to me? |
| KING RICHARD III: | O, I never swear, to give me here. |

Each line ends on the end-of-text token. None of them brings in a second speaker. Juliet's line is still loose. Fine-tuning changed when the model stops, and which speaker it stays on. The lines themselves were still uneven.

## DPO

Fine-tuning had used up the signal in a single correct sentence. The model stopped, and it stayed on the speaker it was given. What still varied was the line itself. King Richard's was tighter than Juliet's. The next loss had to prefer one reply over another.

### Why DPO over PPO

PPO trains a reward model first and then holds it fixed. Every step draws new lines from the model being trained, and the reward model scores them. DPO compares a chosen line and a rejected line that are already in a file. The model writes no new lines while it trains.

These runs were on a MacBook Pro M4 with 24GB of memory. The logits are batch by time by 50,257, and pretraining at batch 64 had already filled that machine. PPO would have added a sampling pass and a reward-model pass on every step. DPO writes the pairs once, Haiku supplying some of the chosen lines, then does one forward pass on each line under the training model and under a frozen copy of the fine-tuned model. After that file exists, training is local and the same run fits a local GPU.

### Which line is better

The fine-tuning file has one sentence per row. DPO needs two replies to the same prompt, and a label for which one is better, so I had to build a new file.

The model already stopped, and it already stayed on the speaker it was given. King Richard's line was tighter than Juliet's. The pairs had to capture that leftover difference.

The prompts are the 287 speaker tags from the fine-tuning training split. That list stays fixed. The pair count changes because of what gets multiplied.

<pre class="mermaid">
flowchart LR
  tags["287 speaker tags"]
  draw["6 lines from the fine-tuned model\ntemperature 0.9, top-k 40"]
  score["preference reward\nform, register, words"]
  pick["highest vs lowest\ndrop ties"]
  few["168 training pairs"]
  write["Haiku writes 3 lines\n861 chosen lines"]
  cross["each chosen line\n× 3 model lines"]
  many["2,325 training pairs"]
  tags --> draw --> score --> pick --> few
  tags --> write --> cross --> many
</pre>

#### Preference reward, both lines from the fine-tuned model

A yes-or-no check tied 284 of the 287 prompts. The preference reward of a line $y$ scores each slip instead. A small miss costs less than a broken line. It only labels the file. The training step does not call it.

$$
r(y) = \mathrm{form}(y) + \mathrm{register}(y) + \mathrm{words}(y)
$$

| Part | When | Score |
|---|---|---|
| form | stops on the end-of-text token | +1 |
| form | misses that token | −2 |
| form | the line is empty | −5 more |
| form | ends in `.`, `?`, or `!` | +0.5 |
| form | misses that mark | −1 |
| form | fewer than two words | −2 |
| form | each word past twenty | −0.15 |
| form | each repeated pair or triple, as in "my lord, my lord" | −0.5 |
| form | commas and semicolons are more than 30% of the words | −1 |
| register | one hit from `thee`, `thou`, `thy`, `thine`, `hath`, `doth`, `dost`, `prithee`, `wherefore`, `whilst`, `naught` | 1, else 0 |
| words | share of the words already in tiny-shakespeare | 0 to 1 |

`It is the king.` stops and ends in a period, so form is 1.5, register is 0, and words is 1: $r = 2.5$. `I have not not not to thy knee.` is form 1.0, register 1, words 1: $r = 3.0$. The worse line wins, and both lines came from the fine-tuned model. Six draws per tag gave 186 pairs, then 168 for training.

#### Chosen line from Haiku

The chosen line in the scored file could be no better than the fine-tuned model's own best draw, and the reward had just ranked a worse draw higher. I needed the preferred line written by something else, and kept short, because this model will not go from `You hear us!` to a full speech in a few hundred steps. Haiku wrote that side once, and the file stayed fixed. Haiku is the teacher and the fine-tuned model is the student. I had the teacher's text, so the teacher's line is $y_w$ and a line from the student is $y_l$. The student is trained to prefer the teacher's line.

Haiku writes three chosen lines for each of the 287 tags. A line is 5 to 15 words and ends in `.`, `?`, or `!`. Each of those lines is paired with three samples from the fine-tuned model. That is $287 \times 3 = 861$ chosen lines and $861 \times 3 = 2{,}583$ pairs, of which 2,325 are for training. The 861 lines cost \$0.16. One pair:

| | Boatswain |
|---|---|
| Chosen, $y_w$ | The storm doth rage with fury against our poor vessel's hull! |
| Rejected, $y_l$ | I must not well. |

The 186 scored pairs are in [`dpo_data.py`](https://github.com/rajathpatel23/post-training-roadmap/blob/main/scratch_nanogpt/dpo_data.py) and [`dpo_scored.csv`](https://github.com/rajathpatel23/post-training-roadmap/blob/main/scratch_nanogpt/datasets/dpo_scored.csv). The 861 Haiku lines are in [`haiku_lines.csv`](https://github.com/rajathpatel23/post-training-roadmap/blob/main/scratch_nanogpt/datasets/haiku_lines.csv). The 2,583 pairs are in [`dpo_gold_data.py`](https://github.com/rajathpatel23/post-training-roadmap/blob/main/scratch_nanogpt/dpo_gold_data.py) and [`dpo_haiku.csv`](https://github.com/rajathpatel23/post-training-roadmap/blob/main/scratch_nanogpt/datasets/dpo_haiku.csv).

### Training

DPO compares the chosen reply with the rejected reply, under the model being trained and under a frozen copy of the fine-tuned model:

$$
\mathcal{L}\_{\mathrm{DPO}} = -\log \sigma \left(\beta \left[ \log \frac{\pi\_\theta(y\_w \mid x)}{\pi\_{\mathrm{ref}}(y\_w \mid x)} - \log \frac{\pi\_\theta(y\_l \mid x)}{\pi\_{\mathrm{ref}}(y\_l \mid x)} \right] \right)
$$

Each $\log \pi(y \mid x)$ is the sum of token log-probabilities over the reply. Pretraining and fine-tuning averaged. This sum is a larger step at the same learning rate. While the training model still matches the frozen copy, the loss is $\ln 2 \approx 0.693$.

Three earlier runs kept the fine-tuning learning rate, `5e-5`. The eval numbers rose and the lines broke. Those runs are in the appendix.

The run I kept uses the Haiku pairs, beta `0.1`, and learning rate `5e-6`. Training is [`train_dpo.py`](https://github.com/rajathpatel23/post-training-roadmap/blob/main/scratch_nanogpt/train_dpo.py), with [`dpo_bpe_gold_lowlr.yaml`](https://github.com/rajathpatel23/post-training-roadmap/blob/main/scratch_nanogpt/configs/train/dpo_bpe_gold_lowlr.yaml). The W&B run is [tinygpt-bpe-dpo-gold-lowlr-v1](https://wandb.ai/rajat-patel/nanogpt-shakespeare/runs/jij4kndg).

### Results

![Loss, reward margin, and preference accuracy from the kept DPO run](/img/posts/tiny-shakespeare/dpo_lowlr_curves.png)

Eval accuracy stayed at 0.955. Eval loss ended at 0.136. The reward margin climbed to 3.33 and stayed there. The train line is a single batch at each logged step, so it jumps around.

| Prompt | Fine-tuned | DPO, learning rate 5e-6 |
|---|---|---|
| ROMEO: | I will not be gone, and see the king. | I dare assure thee for time that Montague and victory! |
| MENENIUS: | You hear us! | I warrant you take good sir, take my father withal. |
| JULIET: | What, that knows not to do not to me? | The moon of Edward's queen's blood of their blood, and Margaret was patient: For triumph is England hast made their heads and slaughter'd me down the Tower. |
| KING RICHARD III: | O, I never swear, to give me here. | But Richard Stanley shall make that Warwick from my heart the world thy words in mine eyes dost Plantagenet is brief. |

Romeo names Montague. Juliet names Edward, Margaret, and the Tower. King Richard names Stanley, Warwick, and Plantagenet. His line still runs on, and some other samples still glue a contraction (`amll'd`, `Iiolanus`). This is the first checkpoint that is clearly ahead of the fine-tuned model. It is `out_dpo_gold_bpe_lowlr/tinygpt_dpo.pt`.

## What changed

Pretraining gave the model the Shakespeare sound and the speaker format, and the turn ran on. Fine-tuning made the turn stop. DPO, once the step size matched a summed loss, preferred a fuller line inside that turn.

Each stage needed data the previous one lacked. The play continues, so a stop had to be cut out as speaker, one sentence, and an end-of-text token. Those sentences are single answers, so DPO needed two answers and a reason to prefer one. Ranking the model's own lines failed once the winner was still a bad line. Haiku wrote the chosen side, in the same short shape.

I would not trust preference accuracy on its own again. On the two broken runs it said the model was learning. The four prompts said otherwise.

## Appendix

### The other DPO runs

Three earlier runs kept the fine-tuning learning rate, `5e-5`. The eval numbers moved the right way. The lines did not.

| Run | Pairs | Beta | Learning rate | Eval loss | Margin | Accuracy | Romeo |
|---|---|---|---|---|---|---|---|
| [Scored pairs](https://wandb.ai/rajat-patel/nanogpt-shakespeare/runs/igvilxk4) | 168 | 0.1 | `5e-5` | 0.395 | 1.02 | 0.833 | And he that I will not so, that is a bloody day. |
| [Haiku pairs](https://wandb.ai/rajat-patel/nanogpt-shakespeare/runs/6tfunf1o) | 2,325 | 0.1 | `5e-5` | 0.015 | 8.80 | 0.996 | I'll imitate yawn castlesel you watch my rage here beheld my son is |
| [Haiku, lower beta](https://wandb.ai/rajat-patel/nanogpt-shakespeare/runs/fpwkzme4) | 2,325 | 0.02 | `5e-5` | 0.082 | 5.18 | 0.977 | Announce kingrophe me |

The configs are [`dpo_bpe.yaml`](https://github.com/rajathpatel23/post-training-roadmap/blob/main/scratch_nanogpt/configs/train/dpo_bpe.yaml), [`dpo_bpe_gold.yaml`](https://github.com/rajathpatel23/post-training-roadmap/blob/main/scratch_nanogpt/configs/train/dpo_bpe_gold.yaml), and [`dpo_bpe_gold_lowbeta.yaml`](https://github.com/rajathpatel23/post-training-roadmap/blob/main/scratch_nanogpt/configs/train/dpo_bpe_gold_lowbeta.yaml).

Preference accuracy asks whether the chosen line outranks the rejected one when the true prefix is filled in. It reached 0.996 on a model whose samples were broken tokens. The free samples were what separated these runs from the one I kept.

### Other checks

A few checks that failed once.

- **Next-token shift.** The target is the next token. `input_ids` drops the last token and `labels` drops the first, the same shift as pretraining. Prompt positions in `labels` are `-100`, except the last prompt position, which predicts the first reply token. An earlier fine-tuning run had this shift backwards, so the loss predicted the current token. That run collapsed into `him him him` and `ROMEO:::::`.
- **End of text.** Pretraining never uses the end-of-text token, so it is appended to the fine-tuning reply.
- **Speaker tags.** A raw line is a wrap, so the speech is rejoined and then cut at the first sentence. A colon is a speaker tag only when the line is short and has no sentence punctuation before it. A scan for colons returned about 2,500 tags, mostly dialogue.
- **Tied scores.** A tied pair is dropped. A score that only checks stopping tied 284 of 287 prompts after fine-tuning had taught stopping.
- **Where the parameters sit.** Most of the 15 million parameters are the vocabulary. The token embedding and the output projection are each 50,257 by 128. The logits are batch by time by vocabulary, and batch 64 with that vocabulary thrashed the machine. Batch 8 fit. Removing layers would leave that tensor the same size. A character model, 4 layers and about 843,000 parameters, did train. Its samples were letter salad, so later stages had nothing readable to judge.

## References

- [post-training-roadmap](https://github.com/rajathpatel23/post-training-roadmap). The code, the configs, and the four CSV tables.
- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), 2019. The GPT-2 byte-pair vocabulary.
- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, and others. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), 2022. A separate reward model, then PPO.
- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017. The PPO update.
- Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), 2023. A loss on a fixed preference file.
