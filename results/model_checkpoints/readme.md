# Model Checkpoints #
Note : All the model training phase is executed using Google Colab T4 GPU.
-- For this project, there are many complexities involved when detecting suicidal risk through multimodal inputs. That's why I have tried multiple transformer models for text classification. Following are the models I finetuned for the suicide specific task.

## 1) DistilBERT ##
-- DistilBERT is a lighter, compressed version of BERT. The reason of using this model is simply performance. It achieves neck-to-neck performance having smaller in parameter size, so it's an ideal choice for text classification.
-- During training, the model was overfitting regardless of the loss functions and other strategies. So, I made several changes in training_parameters, token_size, dataset split in order to minimize the validation loss. After making the changes, I resumed the training progress and saved the best checkpoint based on lower validation loss

## Checkpoints ##

| Epoch | Train Loss | Val Loss  | Accuracy | Precision | Recall  | F1 Score |
|-------|------------|-----------|----------|-----------|---------|----------|
| 18    | 0.101300   | 0.130329  | 0.970497 | 0.959949  | 0.966573 | 0.963250 |
| 19    | 0.114600   | 0.130246  | 0.970497 | 0.958966  | 0.967651 | 0.963289 |
| 20    | 0.101000   | 0.129863  | 0.970756 | 0.963946  | 0.962907 | 0.963426 |



## 2) DistilBERT + CNN ##
-- Combined DistilBERT with CNN model to enhance the prediction accuracy. The tokenization process was done by DistilBERT, feeding tokens to the CNN and then CNN will extract the bi-grams, tri-grams features of words to learn the suicidal/non-suicdial behavior.
-- During training, I have to deal with many problems, made changes in model architecture in order to make inputs compatible for the models.

## Checkpoints ##

| Epoch | Train Loss | Val Loss  | Accuracy | Precision | Recall  | F1 Score |
|-------|------------|-----------|----------|-----------|---------|----------|
| 22    | 0.313300   | 0.313300  | 0.975300 | 0.971400  | 0.966700 | 0.969000 |
| 23    | 0.256300   | 0.256300  | 0.975000 | 0.970000  | 0.967400 | 0.968700 |
| 24    | 0.248800   | 0.248800  | 0.974500 | 0.967300  | 0.969000 | 0.968200 |

---
