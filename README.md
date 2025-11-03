# Phishing Email Detection System
An interactive **cybersecurity monitoring dashboard** built with **Streamlit**, designed to detect anomalies in company email system, suggest response actions, and visualize phishing email trends, response actions, and user impact.

Directly check the demo system with [link](https://cybersecurity-demo-iriswan.streamlit.app)

The app can either:

* Use a local CSV dataset (mocked or exported from an ETL pipeline), or
* Connect directly to **Microsoft 365 mailboxes** through the Graph API.

---

## Features

✅ **Real-time KPIs**

✅ **Phishing email detection**

✅ **Response actions**

✅ **Email reminder to top targeted employees**

✅ **Interactive charts**

✅ **Data filtering**

* Time range (custom or 24h / week / month quick filters)
* Attachment type
* Recipient search
* Threat-only / Safe-only toggle

✅ **Microsoft 365 Integration (optional)**

---

## Architecture Overview

```
Microsoft 365 / CSV Source
        │
        ▼
   ETL / API Layer
   (Python + MSAL + Graph API)
        │
        ▼
   Streamlit Dashboard
  (Visualization & Filtering)
```

---

## Setup & Run

### 1. Clone the repo

```bash
git clone https://github.com/jingjie-wan/Cybersecurity-Demo.git
cd Cybersecurity-Demo
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare your dataset

Option A — use your local CSV:

```
df_result_demo.csv
```

Option B — connect to Microsoft 365 (see below).

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## Screenshot

<img width="2546" height="1352" alt="image" src="https://github.com/user-attachments/assets/2b8ec49e-dddd-4298-afde-3d5ec28c435d" />


---

## Tech Stack

* UI framework (Implementation): **Streamlit**
* Visualization: **Plotly**
* Data Handling: **Pandas / NumPy**
* Modeling (where the prediction labels come from): **BERT (sklearn)**
* Mail Data Access: **MSAL + Microsoft Graph API**
* **Python 3.9+**

---

## 👩‍💻 Author

**Jingjie Wan**
Data Scientist / AI Developer
📧 [Email Me](mailto:iriswan0202@gmail.com)
🔗 [Let's Connect on LinkedIn](https://www.linkedin.com/in/jingjie-wan-86054a257)
