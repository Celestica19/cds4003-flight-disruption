# Implemented Python workflow

```mermaid
flowchart TD
    A["Original flight CSV"] --> B["Deterministic quality checks"]
    B --> C["Date partitions and two day gaps"]
    C --> D["Training period"]
    C --> E["Validation period"]
    C --> F["Untouched test period"]
    D --> G["Preprocessing inside temporal folds"]
    G --> H["Baseline and two model families"]
    H --> I["CV selection and feature ablation"]
    I --> J["Freeze model and alert threshold"]
    E --> J
    J --> K["Final evaluation"]
    F --> K
    K --> L["Metrics tables and figures"]
    J --> M["Persisted model and app inference"]
    M --> N["Public deployment pending"]
```

Missing numerical predictors are imputed using training-only medians. Category encoders, scaling and model fitting are also confined to the training side of each fold. The validation period selects the alert threshold. Test labels do not choose models, hyperparameters, feature variants or thresholds.

This diagram documents the actual Python implementation. It is not a RapidMiner execution screenshot or a RapidMiner process file.
