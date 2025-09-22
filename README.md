<!-- BEGIN README -->

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

1. **Clone this repository**
   ```bash
   git clone https://github.com/frank130845/SIGMA-Refining-Large-Language-Model-Reasoning-via-Sibling-Guided-Monte-Carlo-Augmentation.git
   cd SIGMA-Refining-Large-Language-Model-Reasoning-via-Sibling-Guided-Monte-Carlo-Augmentation
   ```

2. **Install dependencies** (Python ≥ 3.9)
   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:
   ```txt
   torch>=2.0.0
   transformers>=4.40.0
   openai>=1.0.0
   tqdm
   pyyaml
   numpy
   ```

3. **Export OpenAI API configuration**
   ```bash
   export OPENAI_API_BASE="your_base_url"
   export OPENAI_API_KEY="your_api_key"
   ```

4. **Run**
   ```bash
   python main.py
   ```

5. **Outputs**
   - Results and logs are saved under `outputs/`.
   - Edit `config.yaml` to change dataset/model/training params.

---

## 📊 Results
- On **MATH**, a SIGMA-fine-tuned **7B** model reaches **54.92%** accuracy with only **30K** samples, outperforming models trained with **590K** samples.

---

## 📜 Citation
If you find this repo useful, please cite:

```bibtex
@inproceedings{ren2025sigma,
  title={SIGMA: Refining Large Language Model Reasoning via Sibling-Guided Monte Carlo Augmentation},
  author={Ren, Yanwei and Zhang, Haotian and Wu, Fuxiang and Qiu, Jiayan and Huang, Jiaxing and Yu, Baosheng and Liu, Liu},
  booktitle={NeurIPS},
  year={2025}
}
```

---

## ❤️ Acknowledgements
We build on prior work in **LLM reasoning**, **MCTS**, and **chain-of-thought** data generation.

<!-- END README -->
