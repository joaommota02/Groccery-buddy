from google.cloud import vision
from google.oauth2 import service_account
from flask import Flask, request, jsonify
import openai
import json

def detect_text(image_content):
    credentials = service_account.Credentials.from_service_account_file("/var/www/myapp/ok.json")
    client = vision.ImageAnnotatorClient(credentials=credentials)

    image = vision.Image(content=image_content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        return [text.description for text in texts]
    else:
        return "No text found in the image"


def detect_shape(image_content):
    credentials = service_account.Credentials.from_service_account_file("/var/www/myapp/ok.json")
    client = vision.ImageAnnotatorClient(credentials=credentials)

    image = vision.Image(content=image_content)
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations

    if objects:
        return [(obj.name, obj.score) for obj in objects]
    else:
        return []


def detect_receipt(image_content):
    credentials = service_account.Credentials.from_service_account_file("/var/www/myapp/ok.json")

    client = vision.ImageAnnotatorClient(credentials=credentials)

    image = vision.Image(content=image_content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        return [text.description for text in texts]
    else:
        return "No text found in the image"


def detect_macros(image_content):
    credentials = service_account.Credentials.from_service_account_file("/var/www/myapp/ok.json")
    client = vision.ImageAnnotatorClient(credentials=credentials)

    image = vision.Image(content=image_content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        return [text.description for text in texts]
    else:
        return "No text found in the image"


# Initialize Flask app
app = Flask(__name__)
openai.api_key = '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'



# Route for detecting text and suggesting a product
@app.route('/detect-text', methods=['POST'])
def detect_and_suggest():
    image_content = request.data

    detected_text = detect_text(image_content)

    prompt = "Based on the extracted text, suggest a product (10 words max)(always english): '{}'".format(
        detected_text)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=20,
        n=1,
        stop=None,
        temperature=0.6
    )
    suggested_product = response.choices[0].text.strip()

    prompt = "Based on the suggested product, give a small description of the product (50 words max)(always english): '{}'".format(
        suggested_product)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.6
    )
    suggested_description_product = response.choices[0].text.strip()

    prompt = "Based on the extracted text, suggest a location where the product is made (10 words max)(always english): '{}'".format(
        detected_text)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=20,
        n=1,
        stop=None,
        temperature=0.6
    )
    suggested_location_product = response.choices[0].text.strip()

    result = {
        'suggested_product': suggested_product,
        'suggested_description_product': suggested_description_product,
        'suggested_location_product': suggested_location_product
    }

    return jsonify(result)


# Route for detecting shape and suggesting a description
@app.route('/detect-shape', methods=['POST'])
def detect_and_describe_shape():
    image_content = request.data

    detected_shapes = detect_shape(image_content)

    if detected_shapes:
        descriptions = []
        for shape, score in detected_shapes:
            prompt = "Based on the detected shape ({}) with a score of {}, give a small description of the product (30 words max):".format(
                shape, score)
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=50,
                n=1,
                stop=None,
                temperature=0.6
            )
            suggested_description_product = response.choices[0].text.strip()
            descriptions.append({
                'suggested_product': shape,
                'suggested_description_product': suggested_description_product
            })

        return jsonify(descriptions)
    else:
        return "No shapes found in the image."

# Route for detecting text in a receipt and extracting items and price

@app.route('/detect-receipt', methods=['POST'])
def detect_and_suggest_receipt():
    image_content = request.data

    detected_receipt = detect_receipt(image_content)

    prompt = "Based on the extracted text, give me just the name of the products (try to clean the moste you can, to get clear products)  '{}'".format(detected_receipt)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=70,
        n=1,
        stop=None,
        temperature=0.3
    )
    suggested_items = response.choices[0].text.strip()
    
    prompt = "Based on the extracted text, give me only the final price (only the final) '{}'".format(detected_receipt)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=70,
        n=1,
        stop=None,
        temperature=0.3
    )
    suggested_final_price = response.choices[0].text.strip()

    result = {
        'suggested_items': suggested_items,
        'suggested_final_price (€)': suggested_final_price
    }

    return jsonify(result)


# Route for detecting macros and suggesting a description


@app.route('/detect-macros', methods=['POST'])
def detect_and_suggest_macros():
    image_content = request.data

    detected_macros = detect_macros(image_content)

    prompt = "Based on the extracted text, give a description of the macros more focused on the calories, protein, carbs, and fat (in a list format)(in a text way)(without paragraphs)(in a way that everyone can read without effort) (max 50 words): '{}'".format(detected_macros)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.3
    )
    suggested_macros = response.choices[0].text.strip()

    result = {
        'suggested_macros': suggested_macros
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(port=8080, host='0.0.0.0')
