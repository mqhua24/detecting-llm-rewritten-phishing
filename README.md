# detect-llm-rewriten-phishing-emals
This repository contains the code for our research on detecting phishing emails that have been rewritten using Large Language Models (LLMs).
These experiments correspond to a manuscript currently under preparation.

## Overview

Recent LLMs can rewrite phishing emails to make them appear more legitimate and harder to detect.  
In this work, we study how LLM-based rewriting affects phishing email detection systems.
The experiments evaluate three prompt levels and study how LLM rewriting affects phishing detection performance. 
Using a controlled setup, we analyze three key semantic factors in phishing emails and measure their impact on detection results.
We further explore defense strategies for the most impactful factor, including CTA-based risk mitigation methods.

## Experiment Pipeline

The general workflow of the experiments is as follows:

1. Prepare and filter the phishing email dataset
2. Rewrite emails using LLMs with different prompt levels
3. Run phishing detection models on original and rewritten emails
4. Identify three key semantic factors in phishing emails, each with three levels, resulting in 27 factor-level combinations.
5. Use three LLMs to rewrite emails according to prompts designed for each factor-level combination.
6. Run phishing detection models on both original and rewritten emails.
7. Analyze detection performance to determine which semantic factor has the largest impact.
8. Based on the most influential factor, explore corresponding defense strategies (e.g., CTA warning line, CTA short rewrite, and CTA risk override experiments).


## Repository Structure
```text
src/                                # Core implementation
config/                             # Experiment configurations
rewritten_data/config/              # LLM rewriting configuration files; rewritten datasets are withheld due to ethical and security concerns
models_output/                      # Outputs from detection models
llm_detect_data/                    # LLM detection results on rewritten phishing emails
llm_gate_data/                      # CTA-based defense experiment results (CTA-based llm risk override experiments)
llm_cta_data/                       # Placeholder for short-texts and warning-line CTA experiments
                                    # Note: this directory is not included due to ethical considerations
bertscore_output/picture/           # Visualization results and figures
requirements.txt                    # Python dependencies
README.md                           # This file
```

**Main components of the repository include:**

- LLM-based email rewriting

- Phishing detection using existing models

- Data processing and filtering

- Experiment result analysis

- Visualization scripts

## Dataset and Prompts

The original phishing email dataset and the LLM rewriting prompts are not included in this repository due to ethical and security considerations.
These materials contain sensitive content and cannot be publicly shared.

## Detection Models

The experiments use publicly available Hugging Face phishing detection models. 
These models are not included in this repository; please obtain them separately before running the code.

- [ElSlay/BERT-Phishing-Email-Model](https://huggingface.co/ElSlay/BERT-Phishing-Email-Model)
- [ealvaradob/bert-finetuned-phishing](https://huggingface.co/ealvaradob/bert-finetuned-phishing)

## Notes

- Some directories contain experimental outputs or are placeholders for experiments.  
- Certain directories may be empty or not included in the repository due to ethical considerations.  
- This repository is intended for research purposes and may not be fully optimized for production use.