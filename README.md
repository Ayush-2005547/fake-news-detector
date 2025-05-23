# Fake News Detector using BERT

This project presents a machine learning application designed to classify news articles as **Fake** or **Real** using a fine-tuned BERT model. It leverages state-of-the-art NLP techniques and is deployed via Streamlit Cloud for easy public access.

---

## Objective

To develop a high-performance news classification tool that detects misinformation using NLP, specifically BERT-based models.

---

## Abstract

In an age where misinformation spreads rapidly, distinguishing between real and fake news has become critically important. This project uses a pre-trained transformer model, fine-tuned on labeled datasets of news articles, to accurately classify whether a given news input is fake or genuine. The system includes a real-time, user-friendly interface built with Streamlit and hosted on the cloud.

---

## Tools Used

* **Programming Language**: Python 3.x
* **NLP Libraries**: Hugging Face Transformers, NLTK
* **Modeling**: DistilBERT (fine-tuned)
* **Data Handling**: Pandas, NumPy
* **Model Deployment**: Streamlit, Streamlit Cloud
* **Version Control**: Git, GitHub
* **Model File Management**: Git LFS (Large File Storage)

---

## Steps Involved in Building the Project

1. **Dataset Selection**

   * Fake news datasets sourced from Kaggle.

2. **Data Preprocessing**

   * Tokenization and normalization using Hugging Face's tokenizer.

3. **Model Training**

   * DistilBERT fine-tuned on the labeled dataset (fake/real).

4. **Model Evaluation**

   * Accuracy, Precision, Recall, and F1-score computed.

5. **Building the Interface**

   * Interactive frontend using Streamlit.
   * Accepts user input and displays predictions in real time.

6. **Deployment**

   * Deployed publicly via Streamlit Cloud.

---

## Setup Instructions

1. **Clone the Repository**

```bash
git clone [https://github.com/yourusername/fake-news-detector-bert](https://github.com/Ayush-2005547/fake-news-detector).git
cd fake-news-detector
```

2. **Create and Activate Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the App Locally**

```bash
streamlit run app.py
```

---

## Deployment

The application is live and accessible at:
**Streamlit Cloud URL**: (https://fakenews-bert-detector.streamlit.app/)

---

## Repository Structure

```
.
├── app.py                 # Streamlit app script
├── requirements.txt       # Dependencies
├── model.safetensors      # Trained BERT model
├── config.json            # Model configuration
├── .gitattributes         # Git LFS config
├── README.md              # Project documentation
```

---

## Conclusion

This project demonstrates a practical application of NLP and deep learning for media verification. By utilizing DistilBERT and Streamlit, it delivers a scalable and user-friendly tool to detect fake news. This can be a valuable addition to any data science portfolio and serves as a great conversation starter in interviews.

---

## Author

**Ayush Ahirwar**
GitHub: https://github.com/Ayush-2005547

---

## License

This project is licensed under the MIT License.

