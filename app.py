import os
import pickle
import warnings

import pandas as pd
from flask import Flask, render_template_string, request

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ---------------------------------------------------------------- model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Decision_Tree_Algorithm.pkl")
try:
    import joblib
    model = joblib.load(MODEL_PATH)
except Exception:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

FEATURES = [
    "person_age", "person_gender", "person_education", "person_income",
    "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score", "previous_loan_defaults_on_file",
]

# Label-encoded categories (alphabetical order, same as sklearn LabelEncoder).
# If your notebook used different codes, change the numbers here.
GENDER = {"Female": 0, "Male": 1}
EDUCATION = {"Associate": 0, "Bachelor": 1, "Doctorate": 2, "High School": 3, "Master": 4}
HOME = {"Mortgage": 0, "Other": 1, "Own": 2, "Rent": 3}
INTENT = {"Debt Consolidation": 0, "Education": 1, "Home Improvement": 2,
          "Medical": 3, "Personal": 4, "Venture": 5}
DEFAULTS = {"No": 0, "Yes": 1}

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Loan Approval Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --c1:#0f2027; --c2:#203a43; --c3:#2c5364;
    --teal:#00c9a7; --blue:#4facfe; --violet:#8e2de2; --gold:#f9d423; --pink:#ff4b91;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    font-family:'Poppins',sans-serif;min-height:100vh;color:#eaf6ff;
    background:linear-gradient(135deg,var(--c1),var(--c2) 45%,var(--c3));
    background-attachment:fixed;padding:32px 16px 48px;position:relative;overflow-x:hidden;
  }
  body::before,body::after{content:"";position:fixed;border-radius:50%;filter:blur(90px);opacity:.35;z-index:0}
  body::before{width:420px;height:420px;background:var(--violet);top:-120px;left:-100px}
  body::after{width:460px;height:460px;background:var(--teal);bottom:-160px;right:-120px}
  .wrap{position:relative;z-index:1;max-width:980px;margin:0 auto}
  header{text-align:center;margin-bottom:28px}
  .badge{display:inline-block;padding:6px 16px;border-radius:30px;font-size:.78rem;letter-spacing:1.5px;
    text-transform:uppercase;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2)}
  h1{font-size:clamp(1.8rem,4.5vw,3rem);font-weight:700;margin:14px 0 8px;
    background:linear-gradient(90deg,var(--teal),var(--blue),var(--violet),var(--pink));
    -webkit-background-clip:text;background-clip:text;color:transparent}
  header p{font-weight:300;opacity:.85;max-width:620px;margin:0 auto}
  .card{
    background:rgba(255,255,255,.07);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
    border:1px solid rgba(255,255,255,.15);border-radius:24px;padding:30px;
    box-shadow:0 20px 50px rgba(0,0,0,.35);
  }
  .section{margin-bottom:26px}
  .section h2{font-size:1rem;font-weight:600;margin-bottom:14px;display:flex;align-items:center;gap:10px}
  .section h2 span{width:34px;height:34px;border-radius:10px;display:grid;place-items:center;font-size:1rem;
    background:linear-gradient(135deg,var(--teal),var(--blue))}
  .section:nth-of-type(2) h2 span{background:linear-gradient(135deg,var(--violet),var(--pink))}
  .section:nth-of-type(3) h2 span{background:linear-gradient(135deg,var(--gold),var(--pink))}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px}
  label{display:block;font-size:.78rem;font-weight:500;margin-bottom:6px;opacity:.85}
  input,select{
    width:100%;padding:12px 14px;border-radius:12px;font:inherit;font-size:.92rem;color:#fff;
    background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.2);outline:none;transition:.25s;
  }
  select option{color:#102a33}
  input:focus,select:focus{border-color:var(--teal);box-shadow:0 0 0 3px rgba(0,201,167,.3);background:rgba(255,255,255,.14)}
  button{
    width:100%;margin-top:6px;padding:16px;border:none;border-radius:16px;cursor:pointer;
    font:inherit;font-size:1.05rem;font-weight:600;color:#fff;letter-spacing:.5px;
    background:linear-gradient(90deg,var(--teal),var(--blue),var(--violet));background-size:200% 100%;
    transition:.4s;box-shadow:0 10px 25px rgba(79,172,254,.35);
  }
  button:hover{background-position:100% 0;transform:translateY(-2px)}
  .result{margin-top:28px;border-radius:22px;padding:28px;text-align:center;animation:pop .5s ease}
  .result.ok{background:linear-gradient(135deg,rgba(0,201,167,.35),rgba(79,172,254,.3));border:1px solid rgba(0,201,167,.6)}
  .result.no{background:linear-gradient(135deg,rgba(255,75,145,.35),rgba(142,45,226,.3));border:1px solid rgba(255,75,145,.6)}
  .result .icon{font-size:3rem}
  .result h3{font-size:1.7rem;margin:6px 0}
  .result p{font-weight:300;opacity:.95}
  .meter{height:12px;border-radius:12px;background:rgba(255,255,255,.15);margin:16px auto 6px;max-width:420px;overflow:hidden}
  .meter div{height:100%;border-radius:12px;background:linear-gradient(90deg,var(--gold),var(--teal))}
  .result.no .meter div{background:linear-gradient(90deg,var(--gold),var(--pink))}
  .error{margin-top:20px;padding:14px;border-radius:14px;background:rgba(255,75,145,.25);border:1px solid var(--pink)}
  footer{text-align:center;margin-top:22px;font-size:.78rem;opacity:.65}
  @keyframes pop{from{opacity:0;transform:scale(.94)}to{opacity:1;transform:scale(1)}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="badge">Decision Tree Classifier</div>
    <h1>Impact of Loan Approval Classification</h1>
    <p>Enter the applicant's details and the trained model will predict whether the loan is likely to be approved.</p>
  </header>

  <form class="card" method="POST" action="/predict">
    <div class="section">
      <h2><span>👤</span> Personal Details</h2>
      <div class="grid">
        <div><label>Age</label><input type="number" name="person_age" min="18" max="100" required value="{{ v.person_age }}"></div>
        <div><label>Gender</label>
          <select name="person_gender">{% for k in gender %}<option {{ 'selected' if v.person_gender==k }}>{{k}}</option>{% endfor %}</select></div>
        <div><label>Education</label>
          <select name="person_education">{% for k in education %}<option {{ 'selected' if v.person_education==k }}>{{k}}</option>{% endfor %}</select></div>
        <div><label>Annual Income ($)</label><input type="number" step="any" min="0" name="person_income" required value="{{ v.person_income }}"></div>
        <div><label>Employment Experience (years)</label><input type="number" step="any" min="0" name="person_emp_exp" required value="{{ v.person_emp_exp }}"></div>
        <div><label>Home Ownership</label>
          <select name="person_home_ownership">{% for k in home %}<option {{ 'selected' if v.person_home_ownership==k }}>{{k}}</option>{% endfor %}</select></div>
      </div>
    </div>

    <div class="section">
      <h2><span>💰</span> Loan Details</h2>
      <div class="grid">
        <div><label>Loan Amount ($)</label><input type="number" step="any" min="0" name="loan_amnt" required value="{{ v.loan_amnt }}"></div>
        <div><label>Loan Intent</label>
          <select name="loan_intent">{% for k in intent %}<option {{ 'selected' if v.loan_intent==k }}>{{k}}</option>{% endfor %}</select></div>
        <div><label>Interest Rate (%)</label><input type="number" step="any" min="0" name="loan_int_rate" required value="{{ v.loan_int_rate }}"></div>
        <div><label>Loan % of Income (0 – 1)</label><input type="number" step="any" min="0" max="1" name="loan_percent_income" required value="{{ v.loan_percent_income }}"></div>
      </div>
    </div>

    <div class="section">
      <h2><span>📊</span> Credit History</h2>
      <div class="grid">
        <div><label>Credit History Length (years)</label><input type="number" step="any" min="0" name="cb_person_cred_hist_length" required value="{{ v.cb_person_cred_hist_length }}"></div>
        <div><label>Credit Score</label><input type="number" min="300" max="850" name="credit_score" required value="{{ v.credit_score }}"></div>
        <div><label>Previous Loan Defaults on File</label>
          <select name="previous_loan_defaults_on_file">{% for k in defaults %}<option {{ 'selected' if v.previous_loan_defaults_on_file==k }}>{{k}}</option>{% endfor %}</select></div>
      </div>
    </div>

    <button type="submit">🔍 Predict Loan Status</button>

    {% if error %}<div class="error">⚠️ {{ error }}</div>{% endif %}

    {% if result is not none %}
      <div class="result {{ 'ok' if result==1 else 'no' }}">
        <div class="icon">{{ '✅' if result==1 else '❌' }}</div>
        <h3>{{ 'Loan Approved' if result==1 else 'Loan Rejected' }}</h3>
        <p>{{ 'The model predicts this application is likely to be approved.' if result==1 else 'The model predicts this application is likely to be rejected.' }}</p>
        {% if prob is not none %}
          <div class="meter"><div style="width: {{ prob }}%"></div></div>
          <p>Approval probability: <b>{{ prob }}%</b></p>
        {% endif %}
      </div>
    {% endif %}
  </form>
  <footer>Built with Flask &amp; scikit-learn · Deployed on Render</footer>
</div>
</body>
</html>
"""

DEFAULT_VALUES = {
    "person_age": 28, "person_gender": "Male", "person_education": "Bachelor",
    "person_income": 65000, "person_emp_exp": 5, "person_home_ownership": "Rent",
    "loan_amnt": 10000, "loan_intent": "Personal", "loan_int_rate": 11.5,
    "loan_percent_income": 0.15, "cb_person_cred_hist_length": 6,
    "credit_score": 650, "previous_loan_defaults_on_file": "No",
}


class V(dict):
    __getattr__ = dict.get


def render(values, result=None, prob=None, error=None):
    return render_template_string(
        PAGE, v=V(values), result=result, prob=prob, error=error,
        gender=GENDER, education=EDUCATION, home=HOME, intent=INTENT, defaults=DEFAULTS,
    )


@app.route("/")
def home():
    return render(DEFAULT_VALUES)


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form.to_dict()
    try:
        row = {
            "person_age": float(form["person_age"]),
            "person_gender": GENDER[form["person_gender"]],
            "person_education": EDUCATION[form["person_education"]],
            "person_income": float(form["person_income"]),
            "person_emp_exp": float(form["person_emp_exp"]),
            "person_home_ownership": HOME[form["person_home_ownership"]],
            "loan_amnt": float(form["loan_amnt"]),
            "loan_intent": INTENT[form["loan_intent"]],
            "loan_int_rate": float(form["loan_int_rate"]),
            "loan_percent_income": float(form["loan_percent_income"]),
            "cb_person_cred_hist_length": float(form["cb_person_cred_hist_length"]),
            "credit_score": float(form["credit_score"]),
            "previous_loan_defaults_on_file": DEFAULTS[form["previous_loan_defaults_on_file"]],
        }
        X = pd.DataFrame([row], columns=FEATURES)
        pred = int(model.predict(X)[0])
        prob = None
        if hasattr(model, "predict_proba"):
            prob = round(float(model.predict_proba(X)[0][1]) * 100, 1)
        return render(form, result=pred, prob=prob)
    except Exception as e:
        return render(form if form else DEFAULT_VALUES, error=f"Could not predict: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
