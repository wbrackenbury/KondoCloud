import io
import exifread
import json
import re
import celery
import logging
import tensorflow as tf
#import tensorflow_hub as hub

import numpy as np
from memory_profiler import profile

from itertools import combinations
from google.protobuf.json_format import MessageToJson, MessageToDict
from google.cloud import vision
from google.cloud.vision import types

from PIL import Image

from felfinder.utils import jaccard, get_file_type, remove_dups
from felfinder.models import ImageFeats
from felfinder.config import LOCAL_IMG_FEATS

RESNET_SIZE = (224, 224)
ALT_SIZE = (320, 320)
CHOSEN_SIZE = ALT_SIZE
#GLOBAL_RESNET = tf.keras.applications.resnet50.ResNet50(weights='imagenet')

def imbytes_to_imformat(fb, mobilenet = False):

    """
    Takes bytes from an image and turns it into a representation
    we can use for classification
    """

    SIZE = ALT_SIZE if mobilenet else RESNET_SIZE

    NUM_LAST_DIM = 3
    image = Image.open(io.BytesIO(fb))
    img_arr = tf.keras.preprocessing.image.img_to_array(image)
    if img_arr.shape[-1] < NUM_LAST_DIM: # Zero-fill extra channels
        x_dim, y_dim, z_dim = img_arr.shape
        to_fill = NUM_LAST_DIM - z_dim
        zero_filled = tf.zeros((x_dim, y_dim, to_fill))
        img_arr = tf.concat((img_arr, zero_filled), axis=-1)
        #img_arr = tf.tile(img_arr, [1, 1, 3])
    elif img_arr.shape[-1] > NUM_LAST_DIM: # Only use first 3 channels
        img_arr = img_arr[:, :, :NUM_LAST_DIM]
    img_arr = tf.image.resize(img_arr[tf.newaxis, :, :, :], SIZE)

    if mobilenet:
        img_arr = tf.cast(img_arr, tf.uint8)

    return img_arr

def get_exif(fb):
    return json.dumps(Image.open(io.BytesIO(fb))._getexif())

def local_image_feats(fid, user_id, fb):

    try:
        im_arr = imbytes_to_imformat(fb)
        web_labels = local_web_labels(im_arr)
        avg_color = local_avg_color(im_arr)
        exif_data = get_exif(fb)

        ret_obj = ImageFeats(user_id = user_id,
                             id = fid,
                             web_labels = web_labels,
                             avg_color = avg_color,
                             text = str(exif_data))

    except Exception as e:
        logging.error("Failed to convert {} for user {} to image features: {}".\
                      format(fid, user_id, e))

        ret_obj = ImageFeats(user_id = user_id,
                             id = fid,
                             error = True)

    return ret_obj

def local_avg_color(im_arr):

    channels = 3 #Assuming RGB

    unbatch = np.squeeze(im_arr)
    flat = unbatch.reshape((-1, channels))
    avg_c = list(flat.mean(axis=0))
    return ",".join([str(int(c)) for c in avg_c])

def local_web_labels(pred_obj):

    labels = conv_resnet_labels(pred_obj)
    return conv_to_web_labels(labels)

def local_obj_detect(im_arr, model):

    _, pred = model.predict(im_arr)
    return conv_resnet_labels(pred)

def conv_resnet_labels(pred_obj):

    """
    Prediction object looks like:

[
   [
      ('n07753592', 'banana', 0.99229723),
      ('n03532672', 'hook', 0.0014551596),
      ('n03970156', 'plunger', 0.0010738898),
      ('n07753113', 'fig', 0.0009359837) ,
      ('n03109150', 'corkscrew', 0.00028538404)
   ]
]

    And we want to get the labels from each.

    """

    TAKE_THRESH = 0.1

    decoded_pred = tf.keras.applications.imagenet_utils.decode_predictions(pred_obj)
    unbatch = decoded_pred[0]

    #print(unbatch)

    get_pred_obj = lambda x: x[1]
    labels = [get_pred_obj(o) for o in unbatch if o[2] >= TAKE_THRESH]

    return labels

def conv_to_web_labels(labels):

    """
    We're stuck with a rough legacy format--web labels are formatted
    in the database as a string

    '[(label, None)]'

    So we have to convert to that from the labels

    """

    return [(l, None) for l in labels]

def finalize_im_rep(user_id, all_wrappers):


    #uri = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    #mobilenet = False

    # uri = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
    # mobilenet = True

    # model = hub.load(uri)

    # if not mobilenet:
    #     model = model.signatures['default']

    #module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    #detector = hub.load(module_handle).signatures['default']



    resnet = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
    out_layer = resnet.get_layer('avg_pool')
    identity = tf.keras.layers.Lambda(lambda x: x)(out_layer.output)
    pred_layer = resnet.get_layer('predictions')(out_layer.output)

    model = tf.keras.models.Model(inputs = resnet.input,
                                  outputs = [identity, pred_layer])

    for l in model.layers:
        l.trainable = False

    img_feats = []

    for fwrap in all_wrappers:
        try:

            mobilenet = False
            im = imbytes_to_imformat(fwrap.fb, mobilenet)
            # fwrap.im_rep = None
            # output = model(im)

            im_rep, label_preds = model.predict(im)
            fwrap.im_rep = im_rep

            if LOCAL_IMG_FEATS:

                web_labels = local_web_labels(label_preds)
                avg_color = local_avg_color(im)

                fwrap.web_labels = [x[0] for x in web_labels]

                #web_list = remove_dups(output['detection_class_entities'].numpy().tolist())
                #web_labels = str([o.decode('utf-8') for o in web_list][:10])

                ret_obj = ImageFeats(user_id = user_id,
                                     id = fwrap.id,
                                     web_labels = web_labels,
                                     avg_color = avg_color,
                                     text = "")

        except OSError as e:
            ret_obj = ImageFeats(user_id = user_id,
                                 id = fwrap.id,
                                 error = True)

            continue
        except Exception as e:
            logging.error(e)
            ret_obj = ImageFeats(user_id = user_id,
                                 id = fwrap.id,
                                 error = True)

        if LOCAL_IMG_FEATS:
            img_feats.append(ret_obj)

    return img_feats

# Function that extracts the thumbnails for images stored in google drive
# Note that this won't find the google drive files that are photos! We therefore
# won't have image features for them
def get_image_links_google(uid, files):
    links = []
    for f in files:
        if get_file_type(f['name'])[0] == 'image':
            if 'thumbnailLink' in f:
                link = f['thumbnailLink'].replace("=s220","")
                links.append(((str(uid),str(f["id"])) , link))
    return links


def handle_image_annot(api_resp):
    try:
        color_list = api_resp["imagePropertiesAnnotation"]["dominantColors"]["colors"]
    except KeyError:
        return [], ""

    num_colors = len(color_list)
    if num_colors < 1:
        return [], ""

    serialized_colors = []
    str_red_vals = [c['color'].get('red', '0') for c in color_list]
    str_green_vals = [c['color'].get('green', '0') for c in color_list]
    str_blue_vals = [c['color'].get('blue', '0') for c in color_list]

    score_avg = sum([c.get('score', 0) for c in color_list])

    serialized_colors = [(r, g, b, c.get('score', '0'), c.get('pixelFraction', '0')) \
                         for r, g, b, c in zip(str_red_vals,
                                               str_green_vals,
                                               str_blue_vals,
                                               color_list)]
    try:
        avg_red = sum([int(r) * c.get('score', 0) \
                       for r, c in zip(str_red_vals, color_list)]) // score_avg
        avg_green = sum([int(g) * c.get('score', 0) \
                         for g, c in zip(str_green_vals, color_list)]) // score_avg
        avg_blue = sum([int(b) * c.get('score', 0) \
                        for b, c in zip(str_blue_vals, color_list)]) // score_avg
    except ZeroDivisionError:
        return serialized_colors, ""

    avg_color = ','.join([str(int(avg_red)), str(int(avg_green)), str(int(avg_blue))])

    return serialized_colors, avg_color


def handle_web_detect(api_resp):

    try:
        web_entities = api_resp["webDetection"]["webEntities"]
    except KeyError:
        return []

    return [(entity.get("description", None), entity.get("score", None)) \
            for entity in web_entities]

def handle_best_guess(api_resp):

    try:
        web_det = api_resp["webDetection"]["bestGuessLabels"]
    except KeyError:
        return []

    return [(l.get("label", None), l.get("languageCode")) for l in web_det]

def handle_label_annot(api_resp):

    labels = api_resp.get('labelAnnotations', None)
    if labels is None:
        return []

    ret = [(label.get('description', None),
            label.get('score', None),
            label.get('topicality', None)) for label in labels]

    return ret

def handle_full_text(api_resp):

    try:
        init_full_text = api_resp["fullTextAnnotation"]["text"]
    except KeyError:
        return "".encode('utf-8')

    return init_full_text.replace("\n"," ").encode("utf-8",'ignore')


def format_image_req(image_id):

    img_req = {"image":
               {"source":
                {"image_uri": image_id}},
               "features":[
                   {"type":"TEXT_DETECTION"},
                   {"type":"DOCUMENT_TEXT_DETECTION"},
                   {"type":"LABEL_DETECTION"},
                   {"type":"WEB_DETECTION"},
                   {"type":"IMAGE_PROPERTIES"}]}

    return img_req


def get_image_labels(uid, links):


    return []

    #TODO: Actually enable this

    return_img_list = []
    vision_client = vision.ImageAnnotatorClient()
    vision_requests = [format_image_req(image_id = x[1]) for x in links]
    vision_responses = []

    # perform batch requests
    for marker in range(0,len(vision_requests),10):
        print("*************** Performing Request to Vision API ***************")
        response = vision_client.batch_annotate_images(vision_requests[marker:marker+10])
        response = MessageToDict(response)
        vision_responses = vision_responses + response['responses']

    # process the responses
    for index, api_resp in enumerate(vision_responses):
        uid = links[index][0][0]
        file_id = links[index][0][1]

        try:
            serialized_labels = handle_label_annot(api_resp)
            serialized_colors, avg_color = handle_image_annot(api_resp)
            serialized_web_labels  = handle_web_detect(api_resp)
            serialized_best_guess = handle_best_guess(api_resp)
            full_text = handle_full_text(api_resp)
            full_text = full_text.decode('utf-8')

            #HASH FEATURES

            serialized_labels = [label for label in serialized_labels]
            serialized_web_labels = [label for label in serialized_web_labels]
            serialized_best_guess = [label for label in serialized_best_guess]

            ser_lbls = str(serialized_labels)
            ser_colors = str(serialized_colors)
            ser_web_lbls = str(serialized_web_labels)
            best_guess_lbls = str(serialized_best_guess)

            # ser_lbls = sanitize_field(ser_lbls, LONG_FIELD)
            # ser_colors = sanitize_field(ser_colors, LONG_FIELD)
            # ser_web_lbls = sanitize_field(ser_web_lbls, LONG_FIELD)
            # best_guess_lbls = sanitize_field(best_guess_lbls, LONG_FIELD)
            # full_text = sanitize_field(full_text, GIANT_FIELD)
            # avg_color = sanitize_field(avg_color, 31)

            img_label_dict = ImageFeats(user_id=uid, id=file_id, labels=ser_lbls,
                                        colors=ser_colors,
                                        web_labels=ser_web_lbls,
                                        best_guess_labels=str(serialized_best_guess),
                                        text=full_text, avg_color=avg_color, error=False)

            return_img_list.append(img_label_dict)

        except Exception as e:
            print(e)
            print("$"*50)
            print(file_id)
            print("$"*50)
            img_label_dict = dict(user_id=uid, id=file_id, error=True)
            return_img_list.append(img_label_dict)

    return return_img_list

#https://stackoverflow.com/questions/9991933/match-single-quotes-from-python-re
def web_labels_parse(web_labels):

    if web_labels is None:
        return []

    return re.findall("'(\w+)'", web_labels)

def color_str_conv(cvec):

    if cvec is None:
        return [-1, -1, -1]

    converter = lambda x: int(x) if x != "" else 0

    return [converter(it) for it in cvec.split(",")]

def color_sim(cvec1, cvec2):

    if cvec1 == [-1, -1, -1] or cvec2 == [-1, -1, -1] or len(cvec1) != 3 or len(cvec2) != 3:
        return 0.0

    run_diff = 0
    num_colors = 3

    for color in range(num_colors):
        run_diff += abs(cvec1[color] - cvec2[color])

    return (run_diff / num_colors)


def goog_image_simil_feat(uid, simils, image_pairs):

    """
    Calculate similarity features related to images.

    Arguments:
      uid (string): ID of current user we're processings
      simils (list): list of ORM objects from the Simils table
      image_pairs (list[Bool]): lists which values are pairs of images

    Returns:
      simils (list): same ORM objects, but with image similarity features added

    """

    last_im_ID = None
    for s, to_eval in zip(simils, image_pairs):

        # If it is not an image pair, don't bother making the DB query
        if not to_eval:
            continue

        # Because of how similarities are returned, this will likely
        # avoid some DB queries
        if s.file_id_A != last_im_ID:
            im_feat_A = ImageFeats.query.\
                filter_by(user_id=uid).\
                filter_by(id=s.file_id_A).one_or_none()
            last_im_id = s.file_id_A

        im_feat_B = ImageFeats.query.\
                    filter_by(user_id=uid).\
                    filter_by(id=s.file_id_B).one_or_none()

        if not im_feat_A or not im_feat_B or im_feat_A.error or im_feat_B.error:
            continue

        web_labels_A = web_labels_parse(im_feat_A.web_labels)
        web_labels_B = web_labels_parse(im_feat_B.web_labels)

        s.color_sim = color_sim(color_str_conv(im_feat_A.avg_color),
                                     color_str_conv(im_feat_B.avg_color))
        s.obj_sim = jaccard(web_labels_A, web_labels_B)

    return simils
