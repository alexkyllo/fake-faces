import logging
import azure.functions as func
import json
import tempfile
from .predict import predict_image_from_url, predict_image_from_file

# 

def main(req: func.HttpRequest) -> func.HttpResponse:
    image_url = req.params.get('img')
    logging.info('Image URL received: %s', image_url)
    if image_url == None:
        logging.info(req.files.values())
        for input_file in req.files.values():
            logging.info('In Loop!!')
            filename = input_file.filename
            contents = input_file.stream.read()
            logging.info('Filename: %s' % filename)

            image_contents = contents
        logging.info('Out of loop!')
        results = predict_image_from_file(image_contents)
    else:
        results = predict_image_from_url(image_url)

    # TODO: Set Access-Control-Allow-Origin header to hosted location of frontend
    headers = {
        "Content-type": "application/json",
        "Access-Control-Allow-Origin": "*"
    }

    logging.info('Result: %s', results)

    resp = {"result": "Unk"}
    if (results[0] == False):
        resp["result"] = "Not a fake"
    else: resp["result"] = "Fake"

    jsond = json.dumps(resp)
    result = json.loads(jsond)
    return func.HttpResponse(json.dumps(result), headers = headers)
