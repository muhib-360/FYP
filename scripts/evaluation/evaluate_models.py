
def validate_model(checkpoint_path, val_loader, device, criterion):
    """Validate a checkpoint with full metrics including loss"""
    # Initialize model
    model = DistilBERT_CNN().to(device)

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Initialize metrics
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].float().to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask).squeeze()

            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get predictions
            preds = (torch.sigmoid(outputs) > 0.5).cpu().float()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = (np.array(all_labels) == np.array(all_preds)).mean()
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # Print comprehensive results
    print("\n📊 Complete Validation Metrics:")
    print(f"- Loss:      {avg_loss:.4f}")
    print(f"- Accuracy:  {accuracy:.4f}")
    print(f"- Precision: {precision:.4f}")
    print(f"- Recall:    {recall:.4f}")
    print(f"- F1 Score:  {f1:.4f}")
    print("\n🔍 Confusion Matrix:")
    print(f"               Predicted 0 (Non-Suicidal)  Predicted 1 (Suicidal)")
    print(f"Actual 0 (Non-Suicidal) {cm[0,0]:>18} {cm[0,1]:>18}")
    print(f"Actual 1 (Suicidal)     {cm[1,0]:>18} {cm[1,1]:>18}")

# Cell 11: Example Usage with Full Metrics
if __name__ == "__main__":
    # Initialize criterion with same parameters as training
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([config.pos_weight]).to(config.device)
    )


    model = DistilBERT_CNN().to(config.device)

    # Validate specific checkpoint
    epoch_checkpoint = f"{config.save_dir}/checkpoint_epoch.pt"
    validate_model(epoch_checkpoint, val_loader, config.device, criterion)
