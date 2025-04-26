graph LR
    %% Input Pipeline
    A[Raw Text] --> B(Tokenizer)
    B --> C["Token IDs [CLS] I feel... [SEP]"]
    
%%DistilBERT Block
    subgraph DistilBERT["DistilBERT (6-layer)"]
        C --> D[Embeddings]
        D --> E[Transformer Layers]
        E --> F["Contextual Embeddings [batch, 128, 768]"]
    end
    
%% CNN Block
    subgraph CNN["Multi-scale CNN"]
        F --> G["Permute to [batch, 768, 128]"]
        G --> H1["Conv1D (k=2, filters=128)"]
        G --> H2["Conv1D (k=3, filters=128)"]
        G --> H3["Conv1D (k=4, filters=128)"]
        H1 --> I1[MaxPool]
        H2 --> I2[MaxPool]
        H3 --> I3[MaxPool]
        I1 --> J[Concat Features]
        I2 --> J
        I3 --> J
    end
    
  %% Classifier
    J --> K["Combined Features [batch, 384]"] --> L[Dropout] --> M[Linear] --> N[Sigmoid] --> O["Prediction (0-1)"]
    
  %% Styling
    style DistilBERT fill:#e6f3ff,stroke:#3399ff
    style CNN fill:#ffe6e6,stroke:#ff6666
    style O fill:#e6ffe6,stroke:#33cc33
