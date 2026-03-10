from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Home Route
@app.route
def home():
    return render_template("index.html")

@app.route('hello/<name>')
def hello(name):
    return f'<h1> Hello, {name}! </h1><a href="/">Go back</a>'

# Route that handles form submission
@app.route('/greet', methods=['GET', 'POST'])
def greet():
    if request.method == 'POST':
        name = request.form.get('name', 'Guest')
        return redirect(url_for('hello', name=name))
    return render_template('greet.html')

    

if __name__ == "__main__":
    app.run()