from flask import Flask
from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/making-of")
def making_of():
    return render_template('making-of.html')


if __name__ == "__main__":
    app.run(debug=True)
