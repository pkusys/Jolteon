from imageai.Detection import ObjectDetection
import time
import json
import boto3
from boto3.s3.transfer import TransferConfig

from utils import get_suffix

bucketName = 'serverless-bound'

def classify_images(input_adresses, output_adress):
    assert isinstance(input_adresses, list)
    if len(input_adresses) > 0:
        assert isinstance(input_adresses[0], str)
        
    s3_client = boto3.client('s3')
    config = TransferConfig(use_threads=False)
    
    start = int(round(time.time() * 1000)) / 1000.0
    s3_client = boto3.client('s3')
    config = TransferConfig(use_threads=False)
    read_time = 0
    write_time = 0
    comp_time = 0
    
    start_read = int(round(time.time() * 1000)) / 1000.0
    detector = ObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()

    detector.setModelPath('./tiny-yolov3.pt')
    detector.loadModel()
    end_read = int(round(time.time() * 1000)) / 1000.0
    
    read_time += end_read - start_read
    
    for idx, s3_fn in enumerate(input_adresses):
        start_read = int(round(time.time() * 1000)) / 1000.0
        filename = "/tmp/src.jpg"
        f = open(filename, "wb")
        s3_client.download_fileobj(bucketName, s3_fn, f, Config=config)
        f.close()
        end_read = int(round(time.time() * 1000)) / 1000.0
        read_time += end_read - start_read
        
        start_comp = int(round(time.time() * 1000)) / 1000.0
        file_id, chunk_id = get_suffix(s3_fn)
        detection = detector.detectObjectsFromImage(input_image=filename, output_image_path='/tmp/dect_image.jpg',  minimum_percentage_probability=2)

        json_data = json.dumps(detection, indent=4)
        tmp_filename = "/tmp/result.jpg"
        with open(tmp_filename, 'w') as json_file:
            json_file.write(json_data)
        
        end_comp = int(round(time.time() * 1000)) / 1000.0
        comp_time += end_comp - start_comp
        
        start_write = int(round(time.time() * 1000)) / 1000.0
        out_a = output_adress + '_' + file_id + '_' + chunk_id + '.json'
        s3_client.upload_file(tmp_filename, bucketName, out_a, Config=config)
        end_write = int(round(time.time() * 1000)) / 1000.0
        write_time += end_write - start_write
    
    end = int(round(time.time() * 1000)) / 1000.0
    
    return [read_time, comp_time, write_time, end - start]