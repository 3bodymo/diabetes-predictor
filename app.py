from flask import Flask, redirect, url_for, render_template, request
from pid_learn import main, prediction
app = Flask(__name__)

model = main()


@app.route("/", methods=["POST", "GET"])
def dashboard():
    if request.method == "POST":
        tp = float(request.form['tp'])
        pg = float(request.form['pg'])
        dbp = float(request.form['dbp'])
        tst = float(request.form['tst'])
        si = float(request.form['si'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = float(request.form['age'])
        preds = prediction(model, tp, pg, dbp, tst, si, bmi, dpf, age)
        if preds[0] == 0:
            return f"<h3>Negative</h3>"
        elif preds[0] == 1:
            return f"<h3>Positive</h3>"
        else:
            return f"<h3>Unexpected Error!</h3>"

    else:
        return render_template("dashboard.html")


if __name__ == '__main__':
    app.run()
