# SIGMA: Refining Large Language Model Reasoning via Sibling-Guided Monte Carlo Augmentation

<p align="center">
  <img width="661" height="308" alt="overview" src="https://github.com/user-attachments/assets/abb329e6-8693-40b0-babd-bfa32ae62826" />
</p>

<p align="center">
  <img width="693" height="843" alt="architecture" src="https://github.com/user-attachments/assets/2fa2982d-9d05-4445-9ae1-bc2a61f8259c" />
</p>

---

## 🚀 Introduction
**SIGMA (Sibling Guided Monte Carlo Augmentation)** is a novel framework that leverages *sibling nodes* in Monte Carlo Tree Search to refine the reasoning trajectories of large language models (LLMs).  
Instead of discarding non-optimal branches, SIGMA reintegrates them via **critique** and **revision**, significantly boosting reasoning accuracy with far fewer samples.

---

## 📢 News
- 🎉 Our paper has been **accepted as a Poster at NeurIPS 2025**!  

---

## 📖 How to Use

1. **Export your OpenAI API configuration**  
   ```bash
   export OPENAI_API_BASE="your_base_url"
   export OPENAI_API_KEY="your_api_key"
