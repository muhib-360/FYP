```mermaid
%%{init: {'theme': 'neutral', 'fontFamily': 'Arial'}}%%
graph LR
    %% Input Pipeline
    A[Raw Text] --> B(Tokenizer)
    B --> C["Token IDs\n[CLS] I feel... [SEP]"]
    
    %% DistilBERT Block
    subgraph DistilBERT["DistilBERT (6-layer)"]
        C --> D[Embeddings]
        D --> E[Transformer Layers]
        E --> F["Contextual Embeddings\n[batch, 128, 768]"]
    end
    
    %% CNN Block
    subgraph CNN["Multi-scale CNN"]
        F --> G[Permute to\n[batch, 768, 128]]
        G --> H1["Conv1D\n(k=2, filters=128)"]
        G --> H2["Conv1D\n(k=3, filters=128)"]
        G --> H3["Conv1D\n(k=4, filters=128)"]
        H1 --> I1[MaxPool]
        H2 --> I2[MaxPool]
        H3 --> I3[MaxPool]
        I1 --> J[Concat Features]
        I2 --> J
        I3 --> J
    end
    
    %% Classifier
    J --> K["Combined Features\n[batch, 384]"] --> L[Dropout] --> M[Linear] --> N[Sigmoid] --> O["Prediction\n(0-1)"]
    
    %% Styling
    style DistilBERT fill:#e6f3ff,stroke:#3399ff
    style CNN fill:#ffe6e6,stroke:#ff6666
    style O fill:#e6ffe6,stroke:#33cc33
```
