# Intent Classification with BERT on CLINC150

This repository contains an NLP project aimed at classifying user intents using the CLINC150 dataset. The goal is to identify user intent accurately, enabling a better understanding of consumer needs and improving service delivery.

## Project Overview

In this project, a BERT-based model was fine-tuned on the CLINC150 dataset for intent classification. Using PyTorch and leveraging pre-trained models from Hugging Face, the model achieved a testing accuracy of **95%**, demonstrating robust performance in understanding user intent from natural language inputs.

## Dataset

The **CLINC150** dataset is a comprehensive collection of user queries across various intents. It is well-suited for evaluating intent classification systems and contains diverse examples of user interactions.

## Technologies Used

- **Python**
- **PyTorch**: For model training and evaluation.
- **Transformers (Hugging Face)**: For accessing pre-trained BERT model (`bert-base-uncased`) and tokenization.
- **BERT**: Pre-trained deep learning model for natural language processing tasks.
- **CLINC150 Dataset**: Dataset used for fine-tuning and evaluation.

## Project Structure

```
├── data
│   └── clinc150/         # Directory containing the CLINC150 dataset
├── scripts
│   ├── train.py          # Script to train the BERT model
│   ├── evaluate.py       # Script to evaluate the trained model
│   └── preprocess.py     # Script for data preprocessing and tokenization
├── models
│   └── intent_classifier.pt  # Saved PyTorch model after fine-tuning
├── README.md
└── requirements.txt      # Required Python packages
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/intent-classification.git
   cd intent-classification
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download and prepare the CLINC150 dataset:**

   Place the dataset in the `data/clinc150/` directory, or modify the paths in the scripts to point to your dataset location.

## Usage

### Data Preprocessing

Run the preprocessing script to tokenize the dataset and prepare it for training:

```bash
python scripts/preprocess.py
```

### Training

Train the BERT-based intent classifier using the provided script:

```bash
python scripts/train.py
```

Training will fine-tune the `bert-base-uncased` model on the CLINC150 dataset and save the best performing model to `models/intent_classifier.pt`.

### Evaluation

After training, evaluate the model's performance on the test set:

```bash
python scripts/evaluate.py
```

The script will load the fine-tuned model, perform inference on the test data, and report the accuracy. You should see a testing accuracy of around **95%**.

## Results

The fine-tuned BERT model achieved a high degree of accuracy in predicting user intents:

- **Testing Accuracy**: 95%

This performance indicates the model's effectiveness in understanding diverse user queries and accurately classifying their intents, paving the way for improved service delivery based on user needs.

## Conclusion

This project demonstrates the capability of fine-tuning BERT for intent classification using the CLINC150 dataset. It highlights the potential of leveraging pre-trained NLP models and fine-tuning techniques to enhance user interaction systems, leading to more accurate and contextually aware responses.

## Acknowledgements

- [CLINC150 Dataset](https://github.com/stanford-futuredata/Clinc150)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to explore, experiment, and improve upon this intent classification system. Contributions and feedback are welcome!
