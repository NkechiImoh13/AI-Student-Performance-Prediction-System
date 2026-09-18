
from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3, joblib, os
import pandas as pd
from datetime import datetime


app = Flask(__name__)
app.secret_key = "student-performance-project-key"

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "student_performance.db")
MODEL = joblib.load(os.path.join(BASE, "models", "student_performance_random_forest_small.pkl"))
FEATURES = ["attendance","practical","demeanor","presentation","participation","ca","examination"]

def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    conn.execute("""CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        matric_no TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        gender TEXT,
        department TEXT,
        level TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        attendance REAL, practical REAL, demeanor REAL,
        presentation REAL, participation REAL, ca REAL,
        examination REAL, predicted_class INTEGER,
        confidence REAL, early_warning TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(student_id) REFERENCES students(id)
    )""")
    conn.commit()
    conn.close()

def label_prediction(value):
    return {1:"At Risk", 2:"Needs Improvement", 3:"Average",
            4:"Good", 5:"Excellent"}.get(int(value), "Unknown")

def early_warning(vals):
    # Transparent pre-examination rule, separate from the final classifier.
    # Six pre-exam indicators are each scored 0-10.
    avg = sum(vals)/len(vals)
    if avg < 4:
        return "High Risk"
    if avg < 6:
        return "Attention Required"
    return "On Track"

@app.route("/")
def index():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    conn = db()
    total_students = conn.execute("SELECT COUNT(*) c FROM students").fetchone()["c"]
    total_predictions = conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    at_risk = conn.execute(
        "SELECT COUNT(*) c FROM predictions WHERE early_warning IN ('High Risk','Attention Required')"
    ).fetchone()["c"]
    conn.close()
    return render_template("dashboard.html", total_students=total_students,
                           total_predictions=total_predictions, at_risk=at_risk)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":


        if (request.form.get("username") == os.environ.get("ADMIN_USERNAME")
                and request.form.get("password") == os.environ.get("ADMIN_PASSWORD")):
            session["logged_in"] = True
            return redirect(url_for("index"))

        return render_template("login.html", error="Invalid username or password.")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/students")
def students():
    conn = db()
    rows = conn.execute("SELECT * FROM students ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("students.html", students=rows)
@app.route("/students/delete/<int:student_id>", methods=["POST"])
def delete_student(student_id):
    conn = db()
    conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
    conn.commit()
    conn.close()

    return redirect(url_for("students"))

@app.route("/students/add", methods=["GET","POST"])
def add_student():
    if request.method == "POST":
        conn = db()
        try:
            conn.execute("""INSERT INTO students
                (matric_no,name,gender,department,level) VALUES (?,?,?,?,?)""",
                (request.form["matric_no"], request.form["name"],
                 request.form.get("gender"), request.form.get("department"),
                 request.form.get("level")))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return render_template("student_form.html", error="Matriculation number already exists.")
        conn.close()
        return redirect(url_for("students"))
    return render_template("student_form.html")

@app.route("/predict", methods=["GET","POST"])
def predict():
    conn = db()
    students = conn.execute("SELECT * FROM students ORDER BY name").fetchall()
    result = None
    if request.method == "POST":
        vals = [float(request.form[k]) for k in FEATURES]
        X = pd.DataFrame([vals], columns=["x1","x2","x3","x4","x5","x6","x7"])
        pred = int(MODEL.predict(X)[0])
        confidence = float(max(MODEL.predict_proba(X)[0])) * 100
        warning = early_warning(vals[:6])
        sid = int(request.form["student_id"])
        conn.execute("""INSERT INTO predictions
            (student_id,attendance,practical,demeanor,presentation,participation,ca,
             examination,predicted_class,confidence,early_warning,created_at)
             VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (sid,*vals,pred,confidence,warning,datetime.now().strftime("%Y-%m-%d %H:%M")))
        conn.commit()
        result = {"label":label_prediction(pred), "class":pred,
                  "confidence":confidence, "warning":warning}
    conn.close()
    return render_template("predict.html", students=students, result=result)

@app.route("/reports")
def reports():
    conn = db()
    rows = conn.execute("""SELECT p.*, s.matric_no, s.name, s.department
                           FROM predictions p JOIN students s ON p.student_id=s.id
                           ORDER BY p.id DESC""").fetchall()
    conn.close()
    return render_template("reports.html", predictions=rows)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
