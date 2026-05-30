# mBART Urdu Headline Generator

A fine-tuned **mBART-large-50** model for automatically generating Urdu news headlines from article text, using **LoRA (Low-Rank Adaptation)** for parameter-efficient training.

---

## Overview

This project fine-tunes Facebook's `mbart-large-50` multilingual model on an Urdu news dataset to perform headline generation. Training uses 8-bit quantization and LoRA to make the process efficient even on limited GPU resources. The model is evaluated using ROUGE, BERTScore, and BLEU metrics.

---

## Dataset

The model is trained on an Urdu news dataset in `.ndjson` format, where each line is a JSON object with the following structure:

```json
{"text": "Full article body in Urdu...", "headline": "Corresponding headline in Urdu..."}
```

The dataset is split **90% training / 10% evaluation** with a fixed random seed for reproducibility.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Muskan865/Urdu-Headline-Generator.git
cd Urdu-Headline-Generator
```

### 2. Run in Google Colab (recommended)

This notebook is designed to run on **Google Colab** with a GPU runtime. Mount your Google Drive and update the dataset path in the notebook:

```python
filename = '/content/drive/MyDrive/your-dataset.ndjson'
```

---

## Training

The notebook fine-tunes `facebook/mbart-large-50` with the following configuration:

| Parameter | Value |
|---|---|
| Base model | `facebook/mbart-large-50` |
| Quantization | 8-bit (bitsandbytes) |
| LoRA rank (r) | 8 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj, k_proj, o_proj |
| Batch size | 1 (+ gradient accumulation ×16) |
| Learning rate | 5e-5 |
| Epochs | 20 (with early stopping, patience=5) |
| Source/Target language | Urdu (`ur_PK`) |

---

## Evaluation

The model is evaluated on the held-out 10% split using three metrics:

- **ROUGE** (ROUGE-1, ROUGE-2, ROUGE-L)
- **BERTScore** (Precision, Recall, F1 — language: `ur`)
- **SacreBLEU** (BLEU-4)

> 📊 **Results:**

| Metric | Score |
|---|---|
| ROUGE-1 | 0.1442 |
| ROUGE-2 | 0.0195 |
| ROUGE-L | 0.1437 |
| BERTScore Precision | 0.8181 |
| BERTScore Recall | 0.8099 |
| BERTScore F1 | 0.8133 |
| BLEU-4 | 29.47 |
 
> **Note:** The gap between low ROUGE scores and strong BERTScore reflects a known challenge in morphologically rich languages like Urdu — the model generates semantically accurate headlines that use different surface forms than the reference, which ROUGE penalizes but BERTScore captures.

---

## Inference

To generate a headline for a new Urdu article using the saved checkpoint:

```python
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from peft import PeftModel
import torch

base_model = "facebook/mbart-large-50"
checkpoint = "path/to/your/final_checkpoint"

tokenizer = MBart50TokenizerFast.from_pretrained(base_model)
tokenizer.src_lang = "ur_PK"

model = MBartForConditionalGeneration.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, checkpoint)
model.eval()

text = "آپ کا اردو مضمون یہاں لکھیں..."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

output = model.generate(
    **inputs,
    max_length=64,
    num_beams=5,
    no_repeat_ngram_size=3,
    repetition_penalty=2.5
)

headline = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Headline:", headline)
```

---


## Acknowledgements

- [Facebook/Meta AI](https://huggingface.co/facebook/mbart-large-50) for the `mbart-large-50` model
- [Hugging Face](https://huggingface.co) for the Transformers, PEFT, and Datasets libraries
