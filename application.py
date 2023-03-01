from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import PyPDF2
import cv2
import pytesseract
import pandas as pd
import numpy as np
import logging
from PIL import Image
# from logging.handlers import RotatingFileHandler
#Specifying the path of Tesseract
# pytesseract.pytesseract.tesseract_cmd= r'C:\Program Files\Tesseract-OCR\tesseract.exe'                  #For running locally
# os.environ['TESSDATA_PREFIX']="/usr/bin/tesseract"                                                    #For running in cloud
# pytesseract.pytesseract.tesseract_cmd=r"tesseract"  

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'                                            #Cloud
# Set the TESSDATA_PREFIX environment variable
tessdata_dir_config = '--tessdata-dir "/usr/bin/tesseract/tessdata"'


application = Flask(__name__, template_folder='template',static_folder='styles')
logging.basicConfig(filename='logs/record.log', level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
# application.logger.addHandler(RotatingFileHandler('/opt/python/log/application.log', maxBytes=1024,backupCount=5))
application.logger.warning("App starts")
application.logger.info("info")
# Set the location where uploaded files will be stored
UPLOAD_FOLDER = 'uploads/'
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allow files with the following extensions to be uploaded
ALLOWED_EXTENSIONS = {'pdf','ras', 'xwd', 'bmp', 'jpe', 'jpg', 'jpeg', 'xpm', 'ief', 'pbm', 'tif', 'gif', 'ppm','xbm', 'tiff', 'rgb', 'pgm', 'png', 'pnm'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@application.route('/')
def index():
    return render_template('index.html')

# API endpoint for file upload
@application.route('/upload', methods=['POST'])
def upload_files():
    file_count=0
    filenames = []
    for i in range(5):
        # Check if the file is present in the request
        # print(i, ' file{i}' not in request.files)
        if f'file{i+1}' not in request.files:
            continue

        file = request.files[f'file{i+1}']
        # If no file is selected
        # if file.filename == '':
        #     return f'File {i+1} not selected'
      
        filename = secure_filename(file.filename)
        file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))
        filenames.append(filename)
    if(len(filenames)==0):
        return jsonify({"message": "No files selected"})
    return jsonify({"message": "Files uploaded successfully"})


#Function to preprocess image and returns text from it
def img_to_text(img):
    # Pre-processing: Convert the image to grayscale and apply median blur to reduce noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 3)

    # Pre-processing: Enhance image contrast using adaptive histogram equalization
    equalized = cv2.equalizeHist(median)

    # Pre-processing: Apply image restoration techniques to improve image quality
    restored = cv2.inpaint(equalized, median, 3, cv2.INPAINT_TELEA)

    # Threshold the image to reveal the text
    ret, thresh = cv2.threshold(restored, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    dilation = cv2.dilate(thresh, rect_kernel, iterations=10)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    final_text = ''
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Drawing a rectangle on copied image
        rect = cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Cropping the text block for giving input to OCR
        cropped = gray[y:y + h, x:x + w]
        # Apply OCR on the cropped image
        text = pytesseract.image_to_string(cropped)
        final_text = final_text + '' + text
    return final_text



#Dataframe for reference
data = [['Study Permit Application',"your application has been received"],
        ['Study Permit Application',"Principal Applicant"],
        ['Passport Process Request (PPR)',"We require your passport to finalize processing your application"],
        ['Temporary Resident Visa (TRV)','MULTIPLE'],
        ['Study Permit Approval Letter (LOI)',"letter of introduction"],
        ['Study Permit','INFORMATION DU CLIENT']]

df = pd.DataFrame(data, columns=['doctype', 'keywords'])

#To validate the document input-extracted text, output-document type
def documentValidation(text):
    recommended_doc=None
    for item in df.keywords:
        if(text.find(item) != -1):
            recommended_doc = df[['doctype']][(item==df.keywords)]['doctype'].iloc[0]
    return recommended_doc


def textExtractor(doc, file_type):
    text_extracted = ''
    if doc is not None:
        filetype = doc.filename.split(".")[1]  # getting file type
        if filetype.lower() == 'pdf':
            try:
                # pdfReader = PyPDF2.PdfFileReader(doc)
                pdfReader = PyPDF2.PdfReader(doc)
                # print(pdfReader.numPages)
                for i in range(0, len(pdfReader.pages)):                                             #Fixed issue PyPDF not rendering because reader.numpages is deprecated
                    pageObj = pdfReader.pages[i]                                                     #Fixed issue PyPDF not rendering
                    text_extracted = text_extracted + '   ' + pageObj.extract_text()                    #Fixed issue PyPDF not rendering

                if(len(text_extracted)<50):                                                                  #runs if pdf is scanned
                    return "scanned_pdf_warning"
                # print(text_extracted)
                doc.close()

            except Exception as e:
                # st.warning("File contents are not clear. Please verify and re-upload a good quality file.")
                # st.write(e)
                # print(e)
                application.logger.info("info")
                return "not_clear"

        elif filetype.lower() in ['ras', 'xwd', 'bmp', 'jpe', 'jpg', 'jpeg', 'xpm', 'ief', 'pbm', 'tif', 'gif', 'ppm',
                                  'xbm', 'tiff', 'rgb', 'pgm', 'png', 'pnm']:
            try:
                file_bytes = np.asarray(bytearray(doc.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
                text_extracted = img_to_text(opencv_image)
                print("text_extracted")
                print(text_extracted)
            except Exception as e:
                # st.warning("Image is not clear. Please verify and re-upload a good quality file.")
                print("text exception")
                print(e)
                application.logger.info("info")
                return "not_clear"
        else:
            # st.warning("Please upload a valid file (.pdf, .ras, .xwd, .bmp, .jpe, .jpg, .jpeg, .xpm, .ief, .pbm, .tif, .gif, .ppm, .xbm, .tiff, .rgb, .pgm, .png, .pnm)")
            return "invalid_file"

        if (file_type == 'file1'):               #Study Permit Application
            if text_extracted.find("your application has been received") != -1 or text_extracted.find(
                    "Principal Applicant") != -1:
                return "success"
            else:
                doc_recommended = documentValidation(text_extracted)
                if doc_recommended is None and filetype.lower() != 'pdf':                   #For image files, prints unclear if expected words are not found
                    return "not_clear"
                elif doc_recommended is None:
                    return "not_relevant"
                else:
                    return "relevant-"+doc_recommended

        elif (file_type == 'file2'):           #Passport Process Request (PPR)
            if text_extracted.find("We require your passport to finalize processing your application") != -1:
                return "success"
            else:
                doc_recommended = documentValidation(text_extracted)
                if doc_recommended is None and filetype.lower() != 'pdf':  # For image files, prints unclear if expected words are not found
                    return "not_clear"
                elif doc_recommended is None:
                    return "not_relevant"
                else:
                    return "relevant-"+doc_recommended

        elif (file_type == 'file3'):                 #Temporary Resident Visa (TRV)
            if text_extracted.find("MULTIPLE") != -1:
                return "success"
            else:
                doc_recommended = documentValidation(text_extracted)
                if doc_recommended is None and filetype.lower() != 'pdf':  # For image files, prints unclear if expected words are not found
                    return "not_clear"
                elif doc_recommended is None:
                    return "not_relevant"
                else:
                    return "relevant-"+doc_recommended

        elif (file_type == 'file4'):            #Study Permit Approval Letter (LOI)
            if text_extracted.find("letter of introduction") != -1:
                return "success"
            else:
                doc_recommended = documentValidation(text_extracted)
                if doc_recommended is None and filetype.lower() != 'pdf':  # For image files, prints unclear if expected words are not found
                    return "not_clear"
                elif doc_recommended is None:
                    return "not_relevant"
                else:
                    return "relevant-"+doc_recommended

        elif (file_type == 'file5'):                 #Study Permit
            if text_extracted.find("INFORMATION DU CLIENT") != -1:
                return "success"
            else:
                doc_recommended = documentValidation(text_extracted)
                if doc_recommended is None and filetype.lower() != 'pdf':  # For image files, prints unclear if expected words are not found
                    return "not_clear"
                elif doc_recommended is None:
                    return "not_relevant"
                else:
                    return "relevant-"+doc_recommended
    return ''

# API endpoint for file check
@application.route('/file_check', methods=['POST'])
def check_files():
    if 'file' not in request.files:
        return jsonify({"message": "File not selected"})
    file = request.files['file']
    file_name = file.filename
    file_id = request.form.get('input_id')
    status = textExtractor(file, file_id)
    message = ''
    if(status == "success"):
        message = "success"
    elif(status == "not_clear"):
        message = "File contents are not clear. Please verify and re-upload a good quality file."
    elif(status == "invalid_file"):
        message = "Please  upload a valid file (.pdf, .ras, .xwd, .bmp, .jpe, .jpg, .jpeg, .xpm, .ief, .pbm, .tif, .gif, .ppm, .xbm, .tiff, .rgb, .pgm, .png, .pnm)"
    elif(status == "not_relevant"):
        message = "Looks like this is not a relevant document. Please verify."
    elif(status.startswith("relevant")):
        message ="⚠️Please make sure you uploaded the right document.\nThis document looks like a " + status.split("-")[1]
    elif(status.startswith("scanned_pdf_warning")):
        message="File contents are not clear. Please try with an image format."
    return jsonify({"message": message})

if __name__ == "__main__":
    application.run()