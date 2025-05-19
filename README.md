
# Bug Localization with DualTextCNN

This project implements a deep learning-based bug localization system using a Dual TextCNN model. It takes natural language bug reports and ranks source files based on their likelihood of being related to the bug.

---

## 🔍 Project Overview

The system is designed to support tasks such as:

- Training a dual-path CNN classifier
- Evaluating ranked file predictions using IR metrics (Acc@1, Acc@5, MAP, MRR)

It is built using PyTorch and supports fully configurable training and evaluation via a command-line interface.

---

## 📁 Project Structure

```
TEXTCNN_BUGLOCALIZATION/
│
├── artifacts/
│   ├── glove6B/                            # Pretrained GloVe embeddings
│   ├── test_bugs/                          # Bug report pickle files for testing
│   │   └── sphinx-doc+sphinx_bugs.pkl
│   └── trained_model/                      # Saved model checkpoints
│       └── best_model2.pt
│
├── data/
│   ├── data_builder.py
│   ├── glove_loader.py
│   ├── paired_text_dataset.py
│   └── datasets/
│       └── past_brid2commit/
│           ├── sphinx-doc+sphinx_bugs_brid2commit.pkl
│           └── ...                         # other repos' mappings
│
├── evaluation/
│   └── ranker.py                           # Ranking logic and IR metric 
│
├── models/
│   └── dual_text_cnn.py                    # DualTextCNN model
│
├── training/
│   └── trainer.py                          # Trainer class for training and validation
│
├── main.py                                 # CLI entry point
└── README.md

````

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
````

### 2. Prepare Artifacts and Dataset

Place the following in the `artifact/` folder:

* `glove6B/` — Pretrained GloVe embeddings (link: https://nlp.stanford.edu/projects/glove/)
* `past_brid2commit/` — Pickled bug-to-commit mappings
* `*_indexing/` — Indexed source file folders (collected with: https://github.com/ThreeCirclesK/SWE-Bench-History_BugLoc)
* `test_bugs/` — Test bug pickle files

Place the following in the `dataset/` folder:

* `SWEBench_Custom/` — Indexed files for bug localization. Unzip to use them

---

## 🏋️ Training and Evaluation

### Train and Evaluate:

```bash
python main.py \
  --repodir /path/to/SWEBench_Custom \
  --reponame sphinx-doc+sphinx \
  --model_name artifact/trained_model/best_model.pt
```

### Evaluate Only:

```bash
python main.py \
  --evaluate_only \
  --repodir /path/to/SWEBench_Custom \
  --reponame sphinx-doc+sphinx \
  --model_name artifact/trained_model/best_model.pt
```

---

## 📊 Metrics Reported

The evaluation script computes:

* **Accuracy\@1**
* **Accuracy\@5**
* **Mean Reciprocal Rank (MRR)**
* **Mean Average Precision (MAP)**

These metrics reflect the model’s ability to rank fixed files higher among candidates.

---

## 🧠 Model

The `DualTextCNN` processes bug reports and source files in parallel using separate convolutional pathways. It applies max-pooling and concatenates both outputs before classification.

---


## 📄 License

This project is distributed under the MIT License.

