from email.mime import audio
from flask import Flask, request, url_for, render_template

import decodedFile

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

app.template_folder = 'Templates/'
# app.static_url_path = '/static'


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/about')
def aboutPage():
    return render_template('about.html')


@app.route('/help')
def helpPage():
    return render_template('help.html')

# Translate route

@app.route('/translate', methods=['GET', 'POST'])
def text_to_Text():
    if request.method == 'POST':
        # selected_language = request.form.get('inputLang')
        inputText = request.form.get('inputText')
        ans = decodedFile.EncodeAndDecode(inputText)
        context = {'text':ans,'inputText':inputText}
        return render_template('translate.html',context =context)
    else:
        context = ''
        return render_template('translate.html',context=context)

if __name__ == '__main__':
    app.run(debug=True)