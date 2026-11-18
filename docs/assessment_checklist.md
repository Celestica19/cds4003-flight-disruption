# Assessment coverage and outstanding evidence

This checklist maps the uploaded CDS 4003 brief and the separately pasted RapidMiner rubric. It is a review aid, not a claim that the submission is complete or that marks are guaranteed.

Status meanings: **Implemented** means code exists and relevant software tests passed. **Executed real data** means the original 5,819,079-row dataset has been processed and evidence has been generated. **Student or account action** means an approval, authentic contribution, student-written assessment or account-specific task cannot be evidenced by this package.

## Group assessment

| Requirement | Marks | Package support | Outstanding evidence |
|---|---:|---|---|
| Precise business problem, SMART objectives, stakeholders and ML suitability | 3 | Proposed passenger-support prioritization problem and explicit success criteria in README | Instructor approval and student-authored framing remain required; observed CV and lift results meet the proposed analytical targets |
| Dataset >300,000 rows requested by user; at least 20 relevant columns in brief | 3 | Verified original flight dataset 5,819,079 × 31; runtime checks passed | Counts and source hash verified; instructor approval and prior-coursework confirmation remain student responsibilities |
| Source, licence, search terms, accepted/rejected alternatives, limitations and bias | Included above | Source links, search log and technical limitations | Downloaded release verified; instructor dataset approval remains required |
| Systematic pre/post cleaning, missing values, duplicates, inconsistencies, outliers, data types | 4.5 | Full-file chunked audit; quarantine; duplicate index; clock/domain checks; pipeline imputation | Executed audit: no duplicates or conflicts; all rows retained; numeric airport IDs explicitly unresolved; student must explain decisions |
| EDA and meaningful tested features | 4.5 | Five EDA figures, exact training aggregates, statistics and CV feature ablation | Real-data plots and ablation completed; interpret patterns and business implications in own words |
| At least two appropriate models and baseline; fair comparison; no leakage | 6 | Baseline, logistic SGD and histogram boosting; temporal CV and explicit feature allowlist | Three candidates compared; warning log empty; final model and evidence included; student justification required |
| Systematic tuning, search spaces, before/after results, suitable metrics, error trade-offs | 4.5 | Randomized search on temporal folds; AP, ROC AUC, calibration, workload, uncertainty and cost analysis | Real tuning and holdout metrics included; student-authored technical/business interpretation required |
| Public, usable Streamlit app deployed from same repository | 4.5 | App source, input validation, prediction interface, global explanation and performance display | Real model trained and inference checked; install/test Streamlit, finish repository structure, deploy and verify public URL |
| **Group total** | **30** | Implementation supports all seven stages | **Real analysis evidence complete; student report and public deployment pending** |

## GitHub requirements

| Requirement | Status |
|---|---|
| Title and business problem | Revised README describes purpose, scope and measurable objectives |
| Team members and roles | Team members Celestica19 and brhowm2will-jpg identified; full names pending; brhowm2will-jpg assigned planned EDA review, app testing and setup-documentation checks, with completion evidence pending |
| Dataset source, licence and access instructions | Original archive supplied by user; source and hash verified |
| Folder structure and run instructions | Present |
| requirements.txt | Present; core versions tested; Streamlit runtime not installed here |
| Results summary and live app URL | Results summary included; live deployment URL pending |
| Clean code and comments | Present; software tests plus real-data inference verification passed |
| Numbered notebooks with code and explanations | Six notebooks executed with captured real-data outputs |
| Small real-data sample instead of oversized raw data | Real 1,000-row source sample and 20-row app example included |
| No personal absolute paths, passwords or API keys in project code | Static review performed |
| Regular meaningful commits from every student | Student or account action; no fabricated commit history |
| Public repository or instructor collaborator access | Public repository created at https://github.com/Celestica19/cds4003-flight-disruption; root structure needs final verification |
| Same repository used for Streamlit deployment | Pending deployment |
| Data dictionary and dataset-search evidence | Present |
| AI assistance disclosure in README and report | README disclosure present; student report disclosure still required |

## Individual assessment

| Component | Required length | Marks | Evidence the student must supply |
|---|---|---:|---|
| Part A1 Problem framing and data selection | Within Part A total | 10 | Personal critical reasoning, source-search evidence, rejected alternatives, risk and bias |
| Part A2 Cleaning, EDA and features | Within Part A total | 10 | Why each choice fits the actual data; alternatives and effects on later results |
| Part A3 Models, tuning and evaluation | Within Part A total | 12 | How models work; overfitting and validation; metric limits; evidence-based interpretation |
| Part A4 Deployment and business value | Within Part A total | 8 | Deployment evidence, usefulness, realistic recommendations, monitoring, drift, ethics and costs |
| **Part A total** | **2,000–3,000 words** | **40** | All seven project stages, in the student's own words |
| Part B Critical reflection | **500–800 words** | **20** | Specific actual challenges, decisions, alternatives, lessons, limitations and improvements |
| Part C Contribution statement | **At least 300 words** | **10** | Actual tasks, username, commit/file links, collaboration and consistency with repository history |
| **Individual total** |  | **70** | **Independent student writing required** |

## Milestones and submission

| Checkpoint | Week | Required action |
|---|---:|---|
| Milestone 1 | 10 | Instructor approves business problem before dataset search |
| Milestone 2 | 11 | Instructor validates dataset, search evidence and quality issues |
| Milestone 3 | 12 | Instructor reviews cleaning, EDA and feature engineering |
| Milestone 4 | 13 | Instructor reviews at least two models and initial results |
| Milestone 5 | 14 | Instructor reviews live app and ready repository |
| Final submission | 16 | Submit both deliverables to Blackboard AS_FinalProject |

Group deliverable: one **PDF** per team, both repository and live app links on page one, all seven stages, dataset source/licence, AI tools and uses, references. No group-report word count is specified. Individual deliverable: one **PDF or Word** file per student containing Parts A, B and C. The assignment contributes 40% of the course grade. The brief says students may have to explain and defend their code orally. Complete the supplied identity/declaration fields as required by the instructor. Approval history is not supplied here.

## Separate RapidMiner rubric

| Criterion | Maximum | Equivalent Python evidence available after execution | Additional RapidMiner evidence needed if this rubric applies |
|---|---:|---|---|
| Logical, complete, error-free RapidMiner workflow | 3.6 | Numbered stages and reproducible Python pipeline | Genuine RapidMiner process file, correct operator connections/settings and successful run |
| Accurate cleaning and preparation | 3.6 | Data-quality audit and cleaning outputs | Actual RapidMiner execution with matching treatments and results |
| Thorough justification | 2.4 | Technical decision log and generated counts | Student-written explanation of the operators, parameters, alternatives and observed effects |
| Organization, screenshots, labelled figures and tables | 2.4 | Captioned plots and tables generated from real results | Genuine workflow, configuration and before/after result screenshots, clearly labelled |
| **Total** | **12** | Python evidence supports concepts and observed analysis | **No `.rmp` or RapidMiner screenshots are included** |

The Python brief and RapidMiner rubric conflict in required software. Confirm their applicability; do not claim this Python package alone earns the RapidMiner-specific workflow marks. No synthetic project dataset is supplied. The only artificial records are clearly labelled temporary software-test fixtures.

## Documentation expansion review

The README now links to a detailed technical walkthrough and a separate GitHub update guide. All six notebooks include expanded technical explanations while preserving code cells and their recorded outputs. Ten original analysis figures are retained; a temporal-split figure and two explanatory schematics have been added. These documentation changes do not establish public app execution, instructor approval, personal contributions beyond those stated, or completion of student-authored reports.

Outstanding requirements remain: full team identities and evidenced roles; instructor milestone records; public Streamlit runtime and access checks; the live app URL; independent group/individual writing; and confirmation of the separately supplied RapidMiner rubric. The unresolved airport crosswalk remains a technical limitation.
