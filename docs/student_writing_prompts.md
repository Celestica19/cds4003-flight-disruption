# Student writing and evidence prompts

These are planning prompts, not submission prose. The uploaded brief requires student-authored text in both assessments and permits AI only for coding support. Use your own reasoning and actual results. Do not invent experiences, approvals, metrics, business savings or contributions.

## Group report outline

On page one include project title, required student details, repository URL and live Streamlit URL. Then address each of the seven stages below. The brief gives no group word limit.

1. Business problem: Who makes which decision? Why does it matter? What measurable objective and review capacity were agreed? What is the prediction time? What did the instructor approve?
2. Dataset: What was searched, accepted and rejected? What are the actual downloaded row/column counts, target, source, licence and limitations? Has prior-coursework reuse been checked?
3. Understanding and preparation: Which problems actually occurred? What did each change affect? Why was the selected treatment preferable? What remained unresolved? Show before/after counts and row reconciliation.
4. EDA and features: Which non-obvious training-data patterns matter to the business? Are the comparisons confounded? Which new features were tested and what did the ablation show?
5. Models: How do both models and the baseline work? How were comparisons made fair? What did the temporal validation reveal? Why was the final model selected?
6. Tuning and evaluation: What search spaces and folds were used? What improved or worsened? What do AP, precision, recall, calibration and review-budget metrics mean for this decision? What errors and trade-offs remain?
7. Deployment: How is the app used? How are inputs validated? What output supports which action? What limitations, monitoring and retraining would be needed?

Finish with source/licence details where appropriate, an accurate AI-use disclosure, references and any required appendices. Cite figures and tables in the surrounding explanation. Link to the code and machine-generated evidence instead of copying large raw outputs.

## Part A planning allocation

The following 2,500-word allocation is only a planning aid; the required total is 2,000–3,000 words. Confirm how captions/references are counted with the instructor.

| Stage | Suggested words | Questions for your personal justification |
|---|---:|---|
| Business problem | 250 | Which alternative problem did you consider? Why this prediction time, target and stakeholder? |
| Dataset | 300 | Why this source rather than the rejected candidates? Which biases or limitations matter most? |
| Cleaning | 400 | Which decisions required judgment? Compare plausible imputation/outlier/duplicate treatments using actual evidence. |
| EDA and features | 350 | Which result changed your understanding? Did engineered features help? What explanation is supported rather than merely plausible? |
| Model selection | 400 | Explain behaviour, nonlinearity, regularization, overfitting and the baseline comparison. |
| Tuning and evaluation | 450 | Explain temporal validation, parameter choices, metric limits, errors, uncertainty and business trade-offs. |
| Deployment and business interpretation | 350 | Explain practical usability, monitoring, drift, ethics, costs and realistic recommendations. |
| **Total** | **2,500** | Personal reasoning must go beyond repeating the group report. |

## Part B critical reflection

Write 500–800 words based on your experience. Identify a specific challenge; your initial assumption; evidence that supported or challenged it; alternatives considered; what you changed; consequences; limitations; what you learned; and a feasible next improvement. An implementation limitation is not automatically a personal experience. If you did not perform a task, do not claim that you did.

## Part C contribution statement

Write at least 300 words. Include your GitHub username and role. For every claimed contribution, point to a commit, pull request, notebook or file and explain the change and why it mattered. Describe real collaboration and reviews. Disclose coding assistance accurately. Check that dates and ownership agree with the repository history.

## Evidence to capture after the real run

- Source page and licence; actual source shape and checksum.
- Full workflow or notebook sequence with genuine successful outputs.
- Missingness before/after; duplicate and quarantine counts; invalid value corrections; unresolved issues.
- Labelled EDA figures with captions and business interpretation.
- Exact temporal split/fold dates and tuning search results.
- Baseline/model comparison, ablation and final held-out metrics.
- Confusion matrix, precision–recall curve, calibration, subgroup errors and explanation plot.
- Public app URL, valid prediction, validation feedback and deployed performance display.
- Individual commit/file links and milestone approval evidence.

Do not generate imitation screenshots or use test-fixture figures as empirical evidence. The package creates real plots only after the original data is supplied.
