from flask import Flask, render_template, request
import prediction

app = Flask(__name__)

@app.route("/")
def Home_page():
    return render_template('index.html')

@app.route("/submit", methods=["POST"])
def submit():
    income = float(request.form['income'])
    age = float(request.form["age"])
    room = float(request.form["room"])
    bedroom = float(request.form["bedroom"])
    population = float(request.form["population"])

    ypred = prediction.predict(income, age, room, bedroom, population)
    return render_template('index.html', income=income, age=age, room=room, bedroom=bedroom, population=population, ypred = ypred)

if __name__ == "__main__":
    app.run(debug=True)