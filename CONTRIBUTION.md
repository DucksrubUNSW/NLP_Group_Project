# Contributions

**Team Name:** News Group

## Team Members

### Lachlan Johnston (z5477322)

- Set up the project repository and directory structure
- Downloaded and integrated both datasets (LIAR from HuggingFace, FakeNewsNet PolitiFact CSVs from GitHub)
- Built the data loading pipeline (data_loader.py) including label mapping from 6-class LIAR and binary FakeNewsNet to our 3-class scheme (true, mixed, false)
- Implemented the baseline model (TF-IDF + Logistic Regression)
- Implemented BERT and RoBERTa fine-tuning with early stopping, learning rate scheduling, and train/val/test splits
- Built the model evaluation script (evaluate_models.py) for comparing accuracy, precision, recall, and F1 across all models
- Built the test_models.py script for quick single-headline classification across all three models
- Set up Git LFS for storing trained model weights
- Wrote the project README with setup instructions, model download/training options, API key setup, and CLI usage
- Configured the project structure to match submission requirements (CODE/, MISC/, README.md, CONTRIBUTION.md)
- Set up requirements.txt
- Built the Gradio demo.py file which opens up the demo webpage
- Cleaned up comments and style for the codebase
- Wrote report sections covering datasets (3.1 LIAR, 3.2 FakeNewsNet, 3.3 Combined Dataset), modelling (4.1 Baseline, 4.2 Pre-trained BERT, 4.3 Fine-tuned BERT, 4.4 Fine-tuned RoBERTa), and the Gradio demo (6.2)
- Also helped on Quantitative Results (section 5.1) of the report

### Sebastian Litchfield (z5477323)

- Built the CLI tool for running headline classification with support for multiple models and optional WordNet expansion
- Structured CLI outputs to include predictions, confidence scores, and evidence
- Implemented the LangChain retrieval-based fact-checking system using Google Search to gather external evidence and generate evidence-based verdicts (true, mixed, false)
- Developed the prediction combination logic, combining model predictions with evidence and only allowing overrides when model confidence was low and evidence confidence was high
- Implemented WordNet query expansion, generating alternative search queries using both curated synonyms and WordNet with limits on expansions to reduce noise
- Built the evaluation pipeline comparing base model, LangChain, and LangChain + WordNet, reporting accuracy, precision, recall, macro F1, and confusion matrices, and tracking when retrieval corrected or worsened predictions
- Created a balanced evaluation dataset using stratified sampling with equal representation of false, mixed, and true labels
- Produced per-example prediction reports logging model predictions, evidence verdicts, and confidence scores for qualitative error analysis
- Wrote report sections covering WordNet integration (3.4), quantitative results (5.1), qualitative error analysis (5.2), model comparison (5.3), and the CLI (6.1)

### Jaiden Brocklebank (z5421748)

- Implemented initial LangChain functionality with SerperAPI and Gemini API
- Created Initial CLI to input headline and receive output
- Created initial test suite to test a set of headlines
- Section 1 (introductinon), 2.1, 2.2 (problem), 4.5, (langchain), 7 (conclusion) of the report