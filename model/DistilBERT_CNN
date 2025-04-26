```mermaid
%%{init: {'theme': 'neutral', 'fontFamily': 'Arial', 'gantt': {'barHeight': 20}}}%%
graph LR
    %% Input Pipeline
    A[Raw Text] --> B(Tokenizer)
    B --> C["[CLS] I feel... [SEP]</br><sub>Token IDs: [101, 1045, 3867,...]</sub>"]
    
    %% DistilBERT Block
    subgraph DistilBERT
        C --> D[Embeddings]
        D --> E[6 Transformer Layers]
        E --> F["Contextual Embeddings</br>(batch, 128, 768)"]
    end
    
    %% CNN Block
    subgraph CNN
        F --> G[Permute: (batch, 768, 128)]
        G --> H1[Conv1D (k=2)]
        G --> H2[Conv1D (k=3)]
        G --> H3[Conv1D (k=4)]
        H1 --> I1[MaxPool]
        H2 --> I2[MaxPool]
        H3 --> I3[MaxPool]
        I1 --> J[Concat]
        I2 --> J
        I3 --> J
    end
    
    %% Classifier
    J --> K["Features (batch, 384)"] --> L[Dropout 0.2] --> M[Linear] --> N[Sigmoid] --> O["Prediction [0-1]"]
    
    %% Styling
    style DistilBERT fill:#e6f3ff,stroke:#0066cc
    style CNN fill:#ffe6e6,stroke:#cc0000
    style O fill:#e6ffe6,stroke:#00cc00
    
    %% Annotations
    click DistilBERT "https://huggingface.co/docs/transformers/model_doc/distilbert" "DistilBERT Docs"
    click CNN "https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html" "Conv1D Docs"
```
