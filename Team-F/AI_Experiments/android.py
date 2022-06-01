from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def get_image():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    return render_template('index.html', prediction = 100)

@app.route('/receive', methods=['GET', 'POST'])
def receive():
    if request.methods == 'POST':
        f = request.files['image']
        f.save('./images/' + secure_filename(f.filename))
        files = os.listdir("./images")

        remove_background.remove('./images/' + secure_filename(f.filename))                                             
        predition = model.predict(model_transfer, test_transform, class_names, './images/result.jpg')

        os.remove('./images/' + secure_filename(f.filename))

        return jsonify({"cal_result": predition})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
