from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

# Route for the main dashboard
@app.route('/')
def index():
    return render_template('index.html')


app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Define where uploaded files will be stored

# Temporary list of uploaded files for demonstration
uploaded_files = ["Lease_Agreement.pdf", "slave_lease.pdf"]

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/lessee', methods=['GET', 'POST'])
def lessee():
    if request.method == 'POST':
        # Check if a file is uploaded
        file = request.files.get('file')
        if file and file.filename.endswith('.pdf'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            uploaded_files.append(file.filename)
            return redirect(url_for('contract', filename=file.filename))  # Redirect to contract page with filename
    return render_template('lessee.html', files=uploaded_files)


# Route for the Lessor Mode interface
@app.route('/lessor', methods=['GET', 'POST'])
def lessor():
    if request.method == 'POST':
        # Check if a file is uploaded
        file = request.files.get('file')
        if file and file.filename.endswith('.pdf'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            print("saved")
            uploaded_files.append(file.filename)
        return redirect(url_for('contract',filename=file.filename))
    return render_template('lessor.html', files=uploaded_files)


@app.route('/contract/<filename>')
def contract(filename):
    return render_template('contract.html', filename=filename)  # Pass filename to template

if __name__ == '__main__':
    app.run(debug=True)
