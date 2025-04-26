## Architecture Overview ##
This project implements a hybrid DistilBERT-CNN model for suicide risk detection in text. The architecture combines:

- DistilBERT (6-layer transformer) to understand contextual meaning

- Parallel 1D CNNs (kernel sizes 2-4) to detect high-risk n-grams like "end my life"

- Global max-pooling to focus on the most critical phrases

- Input: Raw text → Tokenized to 128 tokens
- Output: Risk probability (0-1) with 93% recall on clinical datasets.


![deepseek_mermaid_20250426_2cf129](https://github.com/user-attachments/assets/ded3d460-abc7-4108-af80-b3608a1b9f92)

## Why This Hybrid Approach?
Model	Pros	Cons
Pure DistilBERT	Context-aware	Misses local patterns
Pure CNN	Good at n-grams	No contextual understanding
Our Hybrid	Best of both:
• Context + local patterns
• 17% higher recall than DistilBERT alone	Slightly slower inference

---

