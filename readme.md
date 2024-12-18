# Study on the Performance of LSTM in Text Sentiment Multi-Classification on the GoEmotions Dataset

**Authors:** Zhuolun Li (NYU), Haolin Yang (NYU)  
**Contact:** zl5583@nyu.edu, hy2898@nyu.edu  
**Code:** [GitHub Repository](https://github.com/icecreamlun/LSTM_in_Text_Sentiment_Classification)

## Abstract
This study evaluates LSTM networks for multi-class sentiment classification using the GoEmotions dataset. We explore three classification levels (3, 6, and 28 classes), achieving test accuracies of 81.12%, 90.27%, and 87.57%, respectively. Results demonstrate that LSTM outperforms traditional methods, making it highly effective for emotional text analysis.

---

## Introduction
- **Goal:** Sentiment analysis for nuanced emotions in text using LSTM.
- **Dataset:** GoEmotions, with 58,000 Reddit comments, 27 emotion labels, and 1 neutral label.
- **Approach:** Progressive classification tasks (3-class → 6-class → 28-class) with iterative model enhancements.

---

## Dataset Configuration
- **Three-Class:** Negative, Positive, Neutral.
- **Six-Class:** Anger, Fear, Joy, Sadness, Surprise, Neutral.
- **Twenty-Eight-Class:** Full emotion spectrum (admiration, amusement, anger, etc.).
- **Preprocessing:** Tokenization, 10,000-token vocabulary, sequence padding (50 tokens), 300-d embedding layer fine-tuned during training.

---

## Model Architecture
### Base LSTM Model
- **Embedding Layer:** 300-d, initialized randomly.
- **LSTM:** Bidirectional, 2 layers, 128 hidden units, 0.5 dropout.
- **Output:** Fully connected layer for classification.

### Enhancements
- **Six-Class Model:** Added MLP layers, increased hidden units, learning rate, and LSTM layers.
- **28-Class Model:** Introduced multi-head attention, additional LSTM layers, and optimized dropout.

---

## Training Configuration
- **Common Settings:**
  - Batch size: 128
  - Optimizer: Adam
  - Loss function: Cross-entropy
  - Epochs: 30
- **Performance:**
  - 3-Class: 81.12%
  - 6-Class: 90.27%
  - 28-Class: 87.57%

---

## Results
### Comparative Analysis (3-Class)
| Model         | Test Accuracy |
|---------------|---------------|
| LSTM          | 81.12%       |
| MLP           | 65.12%       |
| Naive Bayes   | 64.52%       |
| Decision Tree | 61.87%       |
| KNN           | 53.34%       |

**Key Findings:**
- LSTM significantly outperforms traditional methods, showcasing its ability to capture sequential patterns in text.

---

## Conclusion
LSTM-based approaches are effective for multi-class sentiment classification, demonstrating superior performance over traditional methods. Future work can explore integrating transformer architectures and pre-trained models for further improvements.

---

## References
1. Demszky, D., et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions*.  
2. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*.  
3. Vaswani, A., et al. (2017). *Attention is All You Need*.  
