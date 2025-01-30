from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lessee', methods=['GET', 'POST'])
def lessee():
    if request.method == 'POST':
        # Handle file upload and processing here
        uploaded_file = request.files['file']
        if uploaded_file:
            # Process the uploaded file (placeholder)
            # ... Your AI processing logic will go here ...
            return render_template('lessee.html', processed=True)

    return render_template('lessee.html')

@app.route('/lessor')
def lessor():
    return render_template('lessor.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)