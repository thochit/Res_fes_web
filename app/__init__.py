from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import os
import pandas as pd
from app.function_evalua.model_evalua import *
from app.model3.web import model3

app = Flask(__name__)

# Load model;
dict_model = create_dict()
model1 = Model(dict_model)
model1 = model1.load_weight("app/gen_00065000.pt")
# model = model.cuda()
model1.eval()

model2 = Model(dict_model)
model2 = model2.load_weight("app/gen_basso4_3.pt")
# model = model.cuda()
model2.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/gallery')
def gallery():
    # Read the file details from the CSV file
    file_details = pd.read_csv('/media/thochit/DATA/PythonProject/Resfes/file_details.csv', names=['ID', 'name'])

    # Get the first 12 file IDs and names
    file_ids = file_details['ID'].tolist()[:12]
    file_names = file_details['name'].tolist()[:12]

    # Render the gallery template and pass the file IDs and names to it
    return render_template('gallery.html', file_details=zip(file_ids, file_names))


@app.route('/load-more')
def load_more():
    # Read the file details from the CSV file
    file_details = pd.read_csv('/media/thochit/DATA/PythonProject/Resfes/file_details.csv', names=['ID', 'name'])

    # Get the next 12 file IDs and names
    start_index = int(request.args.get('start_index', 0))
    end_index = start_index + 12
    file_ids = file_details['ID'].tolist()[start_index:end_index]
    file_names = file_details['name'].tolist()[start_index:end_index]

    # Return the file IDs and names as a JSON response
    return {'file_details': list(zip(file_ids, file_names))}


@app.route('/ceramic_restoration', methods=['GET', 'POST'])
def ceramic_restoration():
    if request.method == 'POST':
        # Handle the form submission and process the chosen image
        file_id = request.form.get('file_id')
        file_name = request.form.get('file_name')
        print(file_id)
        # Add your image processing logic here

        # Redirect to a success page or render a template
        return render_template('restoration_gallery.html', file_id=file_id)
    else:
        # Handle the GET request, show the restoration page without an image selected
        return render_template('ceramic_restoration.html')
    
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Handle the form submission
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        # Code for sending the message or saving it to a database
        return redirect('/')
    return render_template('contact.html')

@app.route('/send-message', methods=['POST'])
def send_message():
    # Handle the form submission and send the message
    return redirect(url_for('contact'))

@app.route('/save-image', methods=['POST'])
def save_image():
    if request.method == 'POST':
        # Code to save the image
        return 'Image saved successfully'
    else:
        return 'Invalid request method'

@app.route('/', methods=['POST'])
def upload_image():
    # Input image;
    image1 = request.files['image']
    image1.save("/media/thochit/DATA/PythonProject/Resfes/app/static/images/original.jpg")
    image2 = "/media/thochit/DATA/PythonProject/Resfes/app/static/images/original.jpg"
    image = preprocessing_image(image2)
    model3(image2)

    # Create mask and predict;
    x, mask = create_mask(image, 1)
    # x = x.cuda()
    # mask = mask.cuda()
    x1, x2, offset_flow1 = model1(x, mask)
    x3, x4, offset_flow2 = model2(x, mask)
    # x2 = x2.cuda()
    inpainted_result1 = x2 * mask + x * (1. - mask)
    inpainted_result1 = denormalize(inpainted_result1)
    inpainted_result1 = tensor_to_numpy(inpainted_result1)
    inpainted_result1 = convert_image_org(inpainted_result1)

    inpainted_result2 = x4 * mask + x * (1 - mask)
    inpainted_result2 = denormalize(inpainted_result2)
    inpainted_result2 = tensor_to_numpy(inpainted_result2)
    inpainted_result2 = convert_image_org(inpainted_result2)

    # Convert tensor to numpy;
    x = tensor_to_numpy(x)
    image = image.permute(1, 2, 0)
    # image = image.numpy()

    path_save = r"/media/thochit/DATA/PythonProject/Resfes/app/static/images"
    # Path file;
    # image_org = os.path.join(path_save, "original.jpg")
    image_mask = os.path.join(path_save, "masked.jpg")
    image_res1 = os.path.join(path_save, "output1.jpg")
    image_res2 = os.path.join(path_save, "output2.jpg")
    # Save image;
    # cv.imwrite(filename=image_org, img=image * 255)
    cv.imwrite(filename=image_mask, img=x * 255)
    cv.imwrite(filename=image_res1, img=inpainted_result1 * 255)
    cv.imwrite(filename=image_res2, img=inpainted_result2 * 255)

    return redirect('/result')


@app.route('/result')
def result():
    return render_template('result.html')
