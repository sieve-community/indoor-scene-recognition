from typing import List, Dict
from sieve.types import FrameSingleObject, SingleObject, BoundingBox, Temporal
from sieve.predictors import TemporalPredictor

import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from einops import rearrange

labels = {
    "0": "airport indoor",
    "1": "art studio",
    "2": "auditorium",
    "3": "bakery",
    "4": "bar",
    "5": "bathroom",
    "6": "bedroom",
    "7": "bookstore",
    "8": "bowling",
    "9": "buffet",
    "10": "casino",
    "11": "children's room",
    "12": "indoor church",
    "13": "classroom",
    "14": "cloister",
    "15": "closet",
    "16": "clothing store",
    "17": "computer room",
    "18": "concert hall",
    "19": "corridor",
    "20": "deli",
    "21": "dental office",
    "22": "dining room",
    "23": "elevator",
    "24": "fastfood restaurant",
    "25": "florist",
    "26": "game room",
    "27": "garage",
    "28": "greenhouse",
    "29": "grocery store",
    "30": "gym",
    "31": "hair salon",
    "32": "hospital room",
    "33": "inside bus",
    "34": "inside subway",
    "35": "jewelry shop",
    "36": "kindergarten",
    "37": "kitchen",
    "38": "wet laboratory",
    "39": "laundromat",
    "40": "library",
    "41": "living room",
    "42": "lobby",
    "43": "locker room",
    "44": "mall",
    "45": "meeting room",
    "46": "movie theater",
    "47": "museum",
    "48": "nursery",
    "49": "office",
    "50": "operating room",
    "51": "pantry",
    "52": "indoor pool",
    "53": "prison cell",
    "54": "restaurant",
    "55": "restaurant kitchen",
    "56": "shoe shop",
    "57": "stair case",
    "58": "music studio",
    "59": "subway",
    "60": "toy store",
    "61": "train station",
    "62": "television studio",
    "63": "video store",
    "64": "waiting room",
    "65": "warehouse",
    "66": "wine cellar"
}

class SceneRecognizer(TemporalPredictor):
    def setup(self):
        self.vit_extractor = AutoFeatureExtractor.from_pretrained('vincentclaes/mit-indoor-scenes')
        self.vit_model = AutoModelForImageClassification.from_pretrained('vincentclaes/mit-indoor-scenes')
        self.vit_model.eval()
    
    def predict(self, frame: FrameSingleObject) -> List[SingleObject]:
        frame_number = frame.get_temporal().frame_number
        frame_data = frame.get_temporal().get_array()
        with torch.no_grad():
            inputs = self.vit_extractor(images=frame_data, return_tensors='pt')
            outputs = self.vit_model(**inputs).logits
            outputs = rearrange(outputs, '1 j->j')
            outputs = torch.nn.functional.softmax(outputs, dim=0)
            outputs = outputs.cpu().numpy()
            logit_dict = {labels[str(i)]: float(outputs[i]) for i in range(len(labels))}
            max_key = max(logit_dict, key=logit_dict.get)
        
        return [SingleObject(
            cls='scene',
            temporal=Temporal(
                frame_number=frame_number,
                bounding_box=BoundingBox.from_array([0, 0, frame.width, frame.height]),
                score=logit_dict[max_key],
                scene_type=max_key
            )
        )]
