# Google Refexp (Referring Expressions) dataset

The Google RefExp dataset is a collection of text descriptions of objects in 
images which builds on the publicly available [MS-COCO](http://mscoco.org/) 
dataset. Where the image captions in MS-COCO apply to the entire image, this 
dataset focuses on region specific descriptions --- particularly text 
descriptions that allow one to uniquely identify a single object or region 
within an image.

See more details of the collection of the dataset in this paper: [Generation and Comprehension of Unambiguous Object Descriptions](http://arxiv.org/abs/1511.02283)

## Dataset Format

This dataset contains two files of the 201511 released version of train and
validation set of Google Refexp Dataset: 
1.  google_refexp_train_201511.json
2.  google_refexp_val_201511.json

Each json file contains the following key-value pairs:
-  "annotations": list of object annotations in the Google Refexp dataset. Each
   element in the list represented a MS COCO object instance that contains the 
   following key-value pairs:
   -  "annotation_id": id of the original MS COCO object/annotation id.
   -  "image_id": id of the original MS COCO image that the object belongs to.
   -  "refexps_ids": a list of ids of the referring expressions that describe
      this object.
   -  "region_candidates": a list of [multibox](http://arxiv.org/abs/1312.2249) 
      object proposals for the image that the object belongs. Each element in 
      the list contains the following key-value pairs:
      -  "bounding_box": a list of integers \[x, y, w, h\] of the region proposal.
      -  "predicted_object_name": the object name predicted by multibox.
-  "refexps": a list of collected referring expressions. Each element in this 
   list contains the following key-value pairs:
   -  "refexp_id": the id of the referring expression. It is the same id in the
   "refexp_ids" field of each element of "annotations".
   -  "raw": a string of the raw referring expression
   -  "tokens": the tokenized referring expression.
   -  "parse" (optional): the parse results of an internal parse system. Please
   note that this is not a labeld results by humans. It has three fields:

There is also a [toolbox](https://github.com/mjhucla/Google_Refexp_toolbox) to 
visualize the dataset and evaluate the performance of your methods on this 
dataset. It also provide an easy way for you to 
*align Google Refexp dataset with MS COCO*.

## Citation

If you find this dataset useful in your research, please consider citing:

    @article{mao2015generation,
      title={Generation and Comprehension of Unambiguous Object Descriptions},
      author={Mao, Junhua and Huang, Jonathan and Toshev, Alexander and Camburu, Oana and Yuille, Alan and Murphy, Kevin},
      journal={arXiv preprint arXiv:1511.02283},
      year={2015}
    }
    
If you are using the multibox proposals ("region_candidates" field), please also
consider citing:

    @inproceedings{erhan2014scalable,
      title={Scalable object detection using deep neural networks},
      author={Erhan, Dumitru and Szegedy, Christian and Toshev, Alexander and Anguelov, Dragomir},
      booktitle={CVPR},
      pages={2155--2162},
      year={2014},
    }

## Terms of use:

The annotations in this dataset belong to Google and are licensed under a Creative Commons Attribution 4.0 License (see license.txt file).  See also http://mscoco.org/terms_of_use/ for terms of use for the MS-COCO dataset.
