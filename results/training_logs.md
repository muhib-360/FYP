# Training Logs & Results #
## 1) DistilBERT ##

-- Training Logs

Model Performance Metrics

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1 Score |
|-------|---------------|-----------------|----------|-----------|--------|----------|
| 11    | 0.104400      | 0.132793        | 0.968772 | 0.968599  | 0.959364 | 0.960810 |
| 12    | 0.128000      | 0.131435        | 0.957415 | 0.964848  | 0.961117 | 0.960810 |
| 13    | 0.125400      | 0.129768        | 0.969203 | 0.961406  | 0.961613 | 0.961509 |
| 14    | 0.116800      | 0.130310        | 0.969720 | 0.958887  | 0.965711 | 0.962286 |
| 15    | 0.094100      | 0.129708        | 0.969634 | 0.961050  | 0.963123 | 0.962085 |
| 16    | 0.095400      | 0.129437        | 0.969979 | 0.961679  | 0.963338 | 0.962508 |
| 17    | 0.108300      | 0.131464        | 0.970497 | 0.966544  | 0.959457 | 0.962987 |
| 18    | 0.101300      | 0.130329        | 0.970497 | 0.959949  | 0.966573 | 0.963250 |
| 19    | 0.114600      | 0.130246        | 0.970497 | 0.958966  | 0.967651 | 0.963289 |
| 20    | 0.101000      | 0.129863        | 0.970756 | 0.963946  | 0.962907 | 0.963426 |



## Visualizations ##
![cf matrix](https://github.com/user-attachments/assets/8bea9843-a63f-494c-b24c-7e1c74d610e9)


# 2) DistilBERT + CNN #

- Training Logs
-- Manually resumed from epoch 20 (loaded checkpoint_epoch_20.pt)
Epoch 21/24 [Train]:   0%|          | 0/5127 [00:00<?, ?it/s]<ipython-input-14-8e1ef5fb326c>:68: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast():
Epoch 21/24 [Train]: 100%|██████████| 5127/5127 [31:32<00:00,  2.71it/s]
- Epoch 21/24 [Val]: 100%|██████████| 570/570 [04:29<00:00,  2.12it/s]
 Saved checkpoint for epoch 21
Epoch 22/24 [Train]:   0%|          | 0/5127 [00:00<?, ?it/s]<ipython-input-14-8e1ef5fb326c>:68: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast():
- Epoch 22/24 [Train]: 100%|██████████| 5127/5127 [31:24<00:00,  2.72it/s]
- Epoch 22/24 [Val]: 100%|██████████| 570/570 [04:28<00:00,  2.12it/s]
Saved checkpoint for epoch 22
- Epoch 23/24 [Train]:   0%|          | 0/5127 [00:00<?, ?it/s]<ipython-input-14-8e1ef5fb326c>:68: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast():
- Epoch 23/24 [Train]: 100%|██████████| 5127/5127 [31:26<00:00,  2.72it/s]
- Epoch 23/24 [Val]: 100%|██████████| 570/570 [04:29<00:00,  2.11it/s]
Saved checkpoint for epoch 23
Epoch 24/24 [Train]:   0%|          | 0/5127 [00:00<?, ?it/s]<ipython-input-14-8e1ef5fb326c>:68: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast():
- Epoch 24/24 [Train]: 100%|██████████| 5127/5127 [31:22<00:00,  2.72it/s]
- Epoch 24/24 [Val]: 100%|██████████| 570/570 [04:29<00:00,  2.12it/s]







