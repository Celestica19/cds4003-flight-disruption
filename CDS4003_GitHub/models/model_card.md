# Trained model card

| Field | Recorded value |
|---|---|
| Family | Histogram gradient boosting |
| Feature variant | Base schedule features selected by chronological CV ablation |
| Model training rows | 3,888,115 |
| Tuning rows | 120,000 from training only |
| Validation rows | 923,321 |
| Test rows | 947,202 |
| Test dates | 2015-11-01 through 2015-12-31 |
| Model SHA256 | 47960a072f6176477378e985eb1ec1bb951a6d77d679d4ac7c08988c433b3f11 |
| Target | Cancellation, diversion, or arrival at least 15 minutes late |
| Prediction information | Planned schedule only |
| Training target prevalence | 22.0058% |
| Validation target prevalence | 13.1669% |
| Test target prevalence | 19.2632% |
| Test average precision | 0.266330 |
| Test ROC AUC | 0.617095 |
| Frozen individual threshold | 0.327878 |
| Test alert rate at that threshold | 12.1872% |
| Test precision at that threshold | 30.2173% |
| Test recall at that threshold | 19.1175% |
| Precision among top 10 percent | 30.8921% |
| Recall among top 10 percent | 16.0369% |
| Lift among top 10 percent | 1.603689 |

Intended use: a historical 2015 US passenger-support prioritization demonstration. Discrimination is modest and most disruptions are not captured at a 10 percent review capacity. The observed 1.60-fold lift does not establish causation, savings or fitness for current operations. Higher scores are imperfect probability estimates; inspect the calibration chart and Brier score.

The fixed threshold exceeded its validation workload on the test period. Batch ranking enforces the chosen capacity for a supplied list and uses no outcome labels. It does not improve the model or change the frozen test metrics.

October's numeric airport IDs are retained in a separate namespace because an authoritative crosswalk was unavailable. Validation AP was 0.201744 in September and 0.163327 in October. Both code format and month change together, so the observed difference cannot be attributed solely to the identifier issue.

The feature ablation's mean AP difference was −0.000386 for engineered minus base features. This is a small observed difference; no significance claim is made. Parameter and family selection used only training-period folds. Test labels were not used for tuning, selection or threshold adjustment.

The source is public-domain US DOT data as distributed on Kaggle. Source hash, exact environment and configuration are stored with the artifact. Only load this trusted joblib artifact; never deserialize an untrusted uploaded model.

Status: trained artifact and pure Python inference verified. Streamlit runtime and public deployment are not verified. Students must provide their own interpretation, approvals, contribution evidence and final assessment writing.
