from baseline import LanguagePlusImage
from full_model import LanguagePlusImage_Contrast
from Config import Config
from data import ReferExpressionDataset
from refer import REFER
import json
import numpy as np

if __name__ == "__main__":

    cfg = Config()
    refer = REFER(cfg, train=True)
    refer_val = REFER(cfg, False)

    dataset = ReferExpressionDataset(cfg, refer, split=False, MMI=True)
    test_dataset = ReferExpressionDataset(cfg, refer, split=True, test=True)
    val_dataset = ReferExpressionDataset(cfg, refer_val, MMI=True)
    model = LanguagePlusImage_Contrast(cfg, training=True)

    print("Start Training!")
    checkpt_file = 'checkpoints/full.mdl.checkpoint6'
    model.load_model(checkpt_file)
    total_loss = model.run_training(dataset, val_dataset)

    # checkpt_file = 'checkpoints/default.mdl.checkpoint199'
    # model.load_model(checkpt_file)
    # outputs = model.run_test(test_dataset)
    # comprehension_val_inputs = []
    # comprehension_val_input = dict()


    # for output in outputs:
    #     predicted_bounding_boxes = []
    #     ann_id = output['annID'][0].numpy()[0]
    #     comprehension_val_input['annotation_id'] = ann_id
    #
    #     imageID = refer.Anns[str(ann_id )]['image_id']
    #     region_candidates = refer.Imgs[str(imageID)]['region_candidates']
    #     bboxes = [region_candidate['bounding_box'] for region_candidate in region_candidates]
    #     tmp = []
    #     tmp.append(refer.Anns[str(ann_id )]['bbox'])
    #     tmp[1:] = bboxes[:]
    #     bboxes = tmp
    #     for idx in output['sorted_bboxes_idx']:
    #         predicted_bounding_boxes.append(bboxes[idx])
    #     comprehension_val_input['predicted_bounding_boxes'] = predicted_bounding_boxes
    #     comprehension_val_inputs.append(comprehension_val_input)
    #
    # comprehension_file_name = 'comprehension.json'
    # with open(comprehension_file_name, 'w') as f:
    #     json.dump(comprehension_val_inputs, f)



