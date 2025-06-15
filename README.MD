# 📊 Automate Text Insights App

A lightweight and powerful streamlit application to **automatically analyze raw text data**. It extracts useful patterns through bigram, trigram, and word cloud visualizations — and then derives **actionable insights** using GPT-4o Mini. Finally, users can download a well-formatted **PDF report** summarizing the results.

---

## 🚀 Features

* 📁 Upload your **CSV** or **Excel (XLS/XLSX)** file with a `text` column
* 📊 Visualize:

  * **Top 15 Bigrams**
  * **Top 15 Trigrams**
  * **Word Cloud**
* 🤖 Auto-generate insights powered by **GPT-4o Mini**
* 📝 Download a clean multi-page **PDF report** with visualizations and bullet-point insights

---

## 🧑‍💻 How to Use

1. 👉 Go to the app: **[Click here to open the app](https://automate-text-analysis.streamlit.app/)**
2. 📄 Upload your file — accepted formats:

   * `.csv`, `.xls`, or `.xlsx`
   * Must contain a column named **`text`** (case-sensitive)
3. ✅ Wait for the visualizations and AI insights to generate
4. 💾 Download the PDF report directly from the app

---

## 💻 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/roissyahf/automate-text-analysis.git
cd automate-text-analysis

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```