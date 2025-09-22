# SIGMA: Refining Large Language Model Reasoning via Sibling-Guided Monte Carlo Augmentation

<p align="center">
  <img width="661" height="308" alt="overview" src="https://github.com/user-attachments/assets/abb329e6-8693-40b0-babd-bfa32ae62826" />
</p>

<p align="center">
  <img width="693" height="843" alt="architecture" src="https://github.com/user-attachments/assets/2fa2982d-9d05-4445-9ae1-bc2a61f8259c" />
</p>

---

## 🚀 Introduction
**SIGMA (Sibling Guided Monte Carlo Augmentation)** is a framework that leverages *sibling nodes* in Monte Carlo Tree Search (MCTS) to refine LLM reasoning.  
Instead of discarding non-optimal branches, SIGMA links sibling nodes and performs two-stage refinement (**critique → revision**) to improve the top trajectory with less data.

---

## 📢 News
- 🎉 Our paper has been **accepted as a Poster at NeurIPS 2025**!

---

## 📖 How to Use
🚧 Code is still being uploaded. Full usage instructions will be added soon. 🚧

---

## 📊 Results
- On **MATH**, a SIGMA-fine-tuned **7B** model reaches **54.92%** accuracy with only **30K** samples, outperforming models trained with **590K** samples.


## ❤️ Acknowledgements
We build on prior work in **LLM reasoning**, **MCTS**, and **chain-of-thought** data generation.
