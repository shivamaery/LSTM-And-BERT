# Sequential Modeling and Transformer Fine-Tuning for Hierarchical NLP
### Applied Machine Learning (COMP 551)

## Project Overview
This project focused on advanced Natural Language Processing (NLP) techniques, specifically addressing multi-label hierarchical classification. The work involved building a Long Short-Term Memory (LSTM) network from fundamental mathematical principles and benchmarking its performance against a fine-tuned BERT model.

## My Specific Contributions
* **Custom LSTM Implementation:** Developed a complete LSTM architecture from scratch using PyTorch tensors. This included the manual definition of input, forget, and output gates, cell state updates, and orthogonal weight initialization.
* **BERT Fine-Tuning:** Implemented a Transformer-based pipeline using `bert-base-uncased`. I optimized the model using the AdamW optimizer.
* **Comparative Evaluation Suite:** Developed a systematic benchmarking method using Pandas to compare LSTM and BERT across four key metrics: Accuracy and Macro F1-score for both hierarchical levels. I utilized Pandas styling to highlight top-performing models across metrics.
* **Automated Hyperparameter Tuning:** Integrated **Optuna** to conduct a Bayesian search for the LSTM, optimizing the hidden size, learning rate, and batch size to maximize the macro-averaged F1 score.

## Technical Insights
* **Transformer Superiority:** The comparative analysis highlighted the significant advantage of BERT's pre-trained representations and self-attention mechanism over the sequential nature of LSTMs for capturing complex text dependencies.
* **Model Interpretability:** Utilized BERT's internal attention outputs to facilitate potential analysis of token-weighting during the classification process.
* **Class Imbalance Mitigation:** Applied weighted Cross-Entropy Loss functions to both models to account for significant label distribution skews in the dataset.
