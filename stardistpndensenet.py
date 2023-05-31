# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2018. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

from __future__ import print_function, unicode_literals, absolute_import, division


import sys
import numpy as np
import os
import cytomine
from shapely.geometry import shape, box, Polygon,Point
from shapely import wkt
from glob import glob
from tifffile import imread
from cytomine import Cytomine, models, CytomineJob
from cytomine.models import Annotation, AnnotationTerm, AnnotationCollection, ImageInstanceCollection, Job, User, JobData, Project, ImageInstance, Property
from cytomine.models.ontology import Ontology, OntologyCollection, Term, RelationTerm, TermCollection

from csbdeep.utils import Path, normalize
from stardist.models import StarDist2D

from PIL import Image
import torch
from torchvision.models import DenseNet

# import matplotlib.pyplot as plt
import time
import cv2
import math
import csv

from argparse import ArgumentParser
import json
import logging
import logging.handlers
import shutil

__author__ = "WSH Munirah W Ahmad <wshmunirah@gmail.com>"
__copyright__ = "PN Classification: MFA Fauzi, et al. 2015 (https://doi.org/10.1007/978-3-319-19156-0_17)"
__version__ = "1.0.1"
# Stardist (from Cytomine) followed by PN classification and DenseNet Pytorch (Date created: 19 August 2022)

def run(cyto_job, parameters):
    logging.info("----- Stardist-PN-DenseNet Pytorch v%s -----", __version__)
    logging.info("Entering run(cyto_job=%s, parameters=%s)", cyto_job, parameters)

    job = cyto_job.job
    user = job.userJob
    project = cyto_job.project
    threshold_set=parameters.cytomine_th_set
    roi_type=parameters.cytomine_roi_type
    write_hv=parameters.cytomine_write_hv
    modeltype=parameters.cytomine_model
    area_th=parameters.cytomine_area_th

    terms = TermCollection().fetch_with_filter("project", parameters.cytomine_id_project)
    job.update(status=Job.RUNNING, progress=1, statusComment="Terms collected...")
    print(terms)

    start_time=time.time()

    #Loading pre-trained Stardist model
    # base_path = "{}".format(os.getenv("HOME")) # Mandatory for Singularity
#     base_path = os.getcwd()
#     models_path = os.path.join(base_path,"models/")
#     print(base_path)
#     print(models_path)


    #Stardist H&E model downloaded from https://github.com/mpicbg-csbd/stardist/issues/46
    #Stardist H&E model downloaded from https://drive.switch.ch/index.php/s/LTYaIud7w6lCyuI
#     modelsegment = StarDist2D(None, name='2D_versatile_HE', basedir=models_path)   #use local model file in ~/models/2D_versatile_HE/
    modelsegment = StarDist2D(None, name='2D_versatile_HE', basedir='/models/')

    # ----- load network ----
    if modeltype==1: #3k
        modelname = "/models/3333nuclei_densenet_best_model_100ep.pth"
    elif modeltype==2: #22k
        modelname = "/models/22k_nuclei_densenet21_best_model_100ep.pth"

    
    gpuid = 0

    device = torch.device(gpuid if gpuid!=-2 and torch.cuda.is_available() else 'cpu')

    print("Device: ", device)

    checkpoint = torch.load(modelname, map_location=lambda storage, loc: storage) #load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666

    model = DenseNet(growth_rate=checkpoint["growth_rate"], block_config=checkpoint["block_config"],
                    num_init_features=checkpoint["num_init_features"], bn_size=checkpoint["bn_size"],
                    drop_rate=checkpoint["drop_rate"], num_classes=checkpoint["num_classes"]).to(device)

    model.load_state_dict(checkpoint["model_dict"])
    model.eval()
    print("Model name: ",modelname)
    print(f"Total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")
    # ------------------------


    # print(f"Model successfully loaded! Total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")
    job.update(status=Job.RUNNING, progress=20, statusComment=f"Model successfully loaded! Total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

    #Select images to process
    images = ImageInstanceCollection().fetch_with_filter("project", project.id)       
    list_imgs = []
    if parameters.cytomine_id_images == 'all':
        for image in images:
            list_imgs.append(int(image.id))
    else:
        list_imgs = parameters.cytomine_id_images
        list_imgs2 = list_imgs.split(',')
        
    print('Print list images:', list_imgs2)
    job.update(status=Job.RUNNING, progress=30, statusComment="Images gathered...")

    #Set working path
    working_path = os.path.join("tmp", str(job.id))
   
    if not os.path.exists(working_path):
        logging.info("Creating working directory: %s", working_path)
        os.makedirs(working_path)
    try:

        id_project=project.id   
        output_path = os.path.join(working_path, "classification_results.csv")
        f= open(output_path,"w+")

        f.write("AnnotationID;ImageID;ProjectID;JobID;TermID;UserID;Area;Perimeter;Hue;Value;WKT \n")
        
        #Go over images
        for id_image in list_imgs2:

            print('Current image:', id_image)
            roi_annotations = AnnotationCollection(
                terms=[parameters.cytomine_id_roi_term],
                project=parameters.cytomine_id_project,
                image=id_image, #conn.parameters.cytomine_id_image
                showWKT = True,
                includeAlgo=True, 
            )
            roi_annotations.fetch()
            print(roi_annotations)
            #Go over ROI in this image
            #for roi in conn.monitor(roi_annotations, prefix="Running detection on ROI", period=0.1):
            for roi in roi_annotations:
                #Get Cytomine ROI coordinates for remapping to whole-slide
                #Cytomine cartesian coordinate system, (0,0) is bottom left corner
                print("----------------------------ROI------------------------------")
                roi_geometry = wkt.loads(roi.location)
                print("ROI Geometry from Shapely: {}".format(roi_geometry))
                print("ROI Bounds")
                print(roi_geometry.bounds)
                min_x=roi_geometry.bounds[0]
                min_y=roi_geometry.bounds[1]
                max_x=roi_geometry.bounds[2]
                max_y=roi_geometry.bounds[3]
                #Dump ROI image into local PNG file
                roi_path=os.path.join(working_path,str(roi_annotations.project)+'/'+str(roi_annotations.image)+'/'+str(roi.id))
                roi_png_filename=os.path.join(roi_path+'/'+str(roi.id)+'.png')
                print("roi_png_filename: %s" %roi_png_filename)
                is_algo = User().fetch(roi.user).algo
                roi.dump(dest_pattern=roi_png_filename,mask=True,alpha=not is_algo)
                #roi.dump(dest_pattern=os.path.join(roi_path,"{id}.png"), mask=True, alpha=True)
            
                #Stardist works with TIFF images without alpha channel, flattening PNG alpha mask to TIFF RGB
                im=Image.open(roi_png_filename)
                bg = Image.new("RGB", im.size, (255,255,255))
                bg.paste(im,mask=im.split()[3])
                roi_tif_filename=os.path.join(roi_path+'/'+str(roi.id)+'.tif')
                bg.save(roi_tif_filename,quality=100)
                X_files = sorted(glob(roi_path+'/'+str(roi.id)+'*.tif'))
                X = list(map(imread,X_files))
                n_channel = 3 if X[0].ndim == 3 else X[0].shape[-1]
                axis_norm = (0,1)   # normalize channels independently  (0,1,2) normalize channels jointly
                if n_channel > 1:
                    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

                #Going over ROI images in ROI directory (in our case: one ROI per directory)
                for x in range(0,len(X)):
                    print("------------------- Processing ROI file %d: %s" %(x,roi_tif_filename))
                    img = normalize(X[x], parameters.stardist_norm_perc_low, parameters.stardist_norm_perc_high, axis=axis_norm)
                    n_tiles = modelsegment._guess_n_tiles(img)
                    #Stardist model prediction with thresholds
                    labels, details = modelsegment.predict_instances(img,
                                                              prob_thresh=parameters.stardist_prob_t,
                                                              nms_thresh=parameters.stardist_nms_t,
                                                              n_tiles=n_tiles)
                    print("Number of detected polygons: %d" %len(details['coord']))
                    cytomine_annotations = AnnotationCollection()
                    #Go over detections in this ROI, convert and upload to Cytomine
                    for pos,polygroup in enumerate(details['coord'],start=1):
                        #Converting to Shapely annotation
                        points = list()
                        for i in range(len(polygroup[0])):
                            #Cytomine cartesian coordinate system, (0,0) is bottom left corner
                            #Mapping Stardist polygon detection coordinates to Cytomine ROI in whole slide image
                            x_ratio = (max_x-min_x)/im.size[0]
                            y_ratio = (max_y-min_y)/im.size[1]
                            p = Point(min_x+(polygroup[1][i]*x_ratio),max_y-(polygroup[0][i]*y_ratio))
                            points.append(p)

                        annotation = Polygon(points)
                        area=annotation.area
                        if area > area_th: 
                            #Append to Annotation collection 
                            cytomine_annotations.append(Annotation(location=annotation.wkt,
                                                                   id_image=id_image,#conn.parameters.cytomine_id_image,
                                                                   id_project=parameters.cytomine_id_project,
                                                                   id_terms=[parameters.cytomine_id_cell_term]))
                            print(".",end = '',flush=True)

                    #Send Annotation Collection (for this ROI) to Cytomine server in one http request
                    cytomine_annotations.save()


            roi_annotations = AnnotationCollection()
            roi_annotations.project = project.id
            roi_annotations.image = id_image
            roi_annotations.job = job.id
            roi_annotations.user = user
            roi_annotations.showWKT = True
            roi_annotations.fetch()
            # print(roi_annotations)

            hue_all=[]
            val_all=[]

            job.update(status=Job.RUNNING, progress=60, statusComment="Running classification on image...")

            start_prediction_time=time.time()
            predictions = []
            img_all = []
            pred_all = []
            pred_c0 = 0
            pred_c1 = 0
            pred_c2 = 0
            pred_c3 = 0
            
            
            roi_numel=len(roi_annotations)
            x=range(1,roi_numel)
            increment=np.multiply(10000,x)

            #Go over ROI (nuclei) in this image
            for i, roi in enumerate(roi_annotations):
                
                for inc in increment:
                    if i==inc:
                        shutil.rmtree(roi_path, ignore_errors=True)
                        import gc
                        gc.collect()
                        print("i==", inc)

                roi_geometry = wkt.loads(roi.location)

                # print("----------------------------Cells------------------------------")
                #Dump ROI image into local PNG file
                roi_path=os.path.join(working_path,str(roi_annotations.project)+'/'+str(roi_annotations.image)+'/')
                roi_png_filename=os.path.join(roi_path+str(roi.id)+'.png')

                ## --- ROI types: crop or alpha ---
                if roi_type==1: #alpha
                    roi.dump(dest_pattern=roi_png_filename,mask=True)
                elif roi_type==2: #crop
                    roi.dump(dest_pattern=roi_png_filename)
               
                #Start PN classification
                J = cv2.imread(roi_png_filename,cv2.IMREAD_UNCHANGED)
                [r, c, h]=J.shape
                coord=[c//2,r//2]
                y=(coord[1])
                x=(coord[0])

                if r < c:
                    blocksize=r
                else:
                    blocksize=c
                    
                if (blocksize % 2) != 0:
                    blocksize=blocksize-1
                    
                rr=np.zeros((blocksize,blocksize))
                cc=np.zeros((blocksize,blocksize))

                zz=[*range(1,blocksize+1)]
                rr = np.tile(zz, (blocksize, 1))
                cc = np.tile(zz, (blocksize, 1)).T
                
                halfblocksize=(blocksize//2)
 
                weight2 = 1.0 / np.sqrt((cc - halfblocksize + 0.5)**2 + (rr - halfblocksize + 0.5)**2)
                
                # Convert the RGB image to HSV
                currentblock = J[y-halfblocksize:y+halfblocksize,x-halfblocksize:x+halfblocksize,:] #block from centroid               
                Jhsv = cv2.cvtColor(currentblock, cv2.COLOR_BGR2HSV_FULL) 
                Jhsv = Jhsv/255

                currentblockH=Jhsv[:,:,0]
                currentblockV=1-Jhsv[:,:,2]
                hue=sum(sum(currentblockH*weight2))
                val=sum(sum(currentblockV*weight2))
#                 print("hue:", hue)
#                 print("val:", val)
                hue_all.append(hue)
                val_all.append(val)

                if threshold_set==1:
                    #--- Threshold values (modified-used on v0.1.25 and earlier)---
                    # FP case (positive as negative): Hue 4.886034191808089 Val 14.45894207427296 
                    if hue<5:#mod1<2; mod2<5
                       cellclass=2
                    elif val<15:
                       cellclass=1
                    else:
                        if hue<30 or val>40:
                           cellclass=2
                        else:
                           cellclass=1
                    #--------------------------------------------------------------
                elif threshold_set==2:
                    #--- Threshold values (original-used on v2 from 18 April 2022)---
                    if val>50:
                       cellclass=2
                    else:
                        if hue>70:
                            cellclass=1
                        else:
                            cellclass=2
                    #----------------------------------------------------------------
                elif threshold_set==3:    
                    print("Th set 3 not yet defined.")
                    #----------------------------------------------------------------

                if cellclass==1: #negative

                    id_terms=parameters.cytomine_id_c0_term
                    pred_c0=pred_c0+1

                elif cellclass==2: #positive
#                     
                    #Start WMS DenseNet classification for positive nucleus
                    im = cv2.cvtColor(cv2.imread(roi_png_filename),cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im,(224,224))
                    im = im.reshape(-1,224,224,3)
                    output = np.zeros((0,checkpoint["num_classes"]))
                    arr_out_gpu = torch.from_numpy(im.transpose(0, 3, 1, 2)).type('torch.FloatTensor').to(device)
                    output_batch = model(arr_out_gpu)
                    output_batch = output_batch.detach().cpu().numpy()                
                    output = np.append(output,output_batch,axis=0)
                    pred_labels = np.argmax(output, axis=1)
                    # pred_labels=[pred_labels]
                    pred_all.append(pred_labels)

                    if pred_labels[0]==0:
                        # print("Class 0: Negative")
                        id_terms=parameters.cytomine_id_c0_term
                        pred_c0=pred_c0+1
                        # roi.dump(dest_pattern=os.path.join(roi_path+'Class0/'+str(roi.id)+'.png'),alpha=True)
                    elif pred_labels[0]==1:
                        # print("Class 1: Weak")
                        id_terms=parameters.cytomine_id_c1_term
                        pred_c1=pred_c1+1
                        # roi.dump(dest_pattern=os.path.join(roi_path+'Class1/'+str(roi.id)+'.png'),alpha=True)
                    elif pred_labels[0]==2:
                        # print("Class 2: Moderate")
                        id_terms=parameters.cytomine_id_c2_term
                        pred_c2=pred_c2+1
                        # roi.dump(dest_pattern=os.path.join(roi_path+'Class2/'+str(roi.id)+'.png'),alpha=True)
                    elif pred_labels[0]==3:
                        # print("Class 3: Strong")
                        id_terms=parameters.cytomine_id_c3_term
                        pred_c3=pred_c3+1
                  
                
                cytomine_annotations = AnnotationCollection()
                annotation=roi_geometry
                cytomine_annotations.append(Annotation(location=annotation.wkt,#location=roi_geometry,
                                                       id_image=id_image,#conn.parameters.cytomine_id_image,
                                                       id_project=project.id,
                                                       id_terms=[id_terms]))
                print(".",end = '',flush=True)



                #Send Annotation Collection (for this ROI) to Cytomine server in one http request
                ca = cytomine_annotations.save()

            cytomine_annotations = AnnotationCollection()    
            cytomine_annotations.project = project.id
            cytomine_annotations.image = id_image
            cytomine_annotations.job = job.id
            cytomine_annotations.user = user
            cytomine_annotations.showAlgo = True
            cytomine_annotations.showWKT = True
            cytomine_annotations.showMeta = True
            cytomine_annotations.showGIS = True
            cytomine_annotations.showTerm = True
            cytomine_annotations.annotation = True
            cytomine_annotations.fetch()

            hue_all.reverse()
            val_all.reverse()

            end_prediction_time=time.time()

            # ## --------- WRITE Hue and Value values into CSV and annotation Property -----------
            if write_hv==1:
                job.update(status=Job.RUNNING, progress=80, statusComment="Writing classification results on CSV...")
                for j, annotation in enumerate(cytomine_annotations):
                    if annotation.term!=parameters.cytomine_id_cell_term:
                        f.write("{};{};{};{};{};{};{};{};{};{};{}\n".format(annotation.id,annotation.image,annotation.project,job.id,annotation.term,annotation.user,annotation.area,annotation.perimeter,str(hue_all[j]),str(val_all[j]),annotation.location))
                        Property(Annotation().fetch(annotation.id), key='Hue', value=str(hue_all[j])).save()
                        Property(Annotation().fetch(annotation.id), key='Val', value=str(val_all[j])).save()
                        Property(Annotation().fetch(annotation.id), key='ID', value=str(annotation.id)).save()
            ##---------------------------------------------------------------------------

            start_scoring_time=time.time()

            job.update(status=Job.RUNNING, progress=90, statusComment="Generating scoring for whole-slide image(s)...")
            pred_all=[pred_c0, pred_c1, pred_c2, pred_c3]
            pred_positive_all=[pred_c1, pred_c2, pred_c3]
            print("pred_all:", pred_all)
            im_pred = np.argmax(pred_all)
            print("image prediction:", im_pred)
            pred_total=pred_c0+pred_c1+pred_c2+pred_c3
            print("pred_total:",pred_total)
            pred_positive=pred_c1+pred_c2+pred_c3
            print("pred_positive:",pred_positive)
            print("pred_positive_all:",pred_positive_all)
            print("pred_positive_max:",np.argmax(pred_positive_all))
            pred_positive_100=pred_positive/pred_total*100
            print("pred_positive_100:",pred_positive_100)

            if pred_positive_100 == 0:
                proportion_score = 0
            elif pred_positive_100 < 1:
                proportion_score = 1
            elif pred_positive_100 >= 1 and pred_positive_100 <= 10:
                proportion_score = 2
            elif pred_positive_100 > 10 and pred_positive_100 <= 33:
                proportion_score = 3
            elif pred_positive_100 > 33 and pred_positive_100 <= 66:
                proportion_score = 4
            elif pred_positive_100 > 66:
                proportion_score = 5

            if pred_positive_100 == 0:
                intensity_score = 0
            elif im_pred == 0:
                intensity_score = np.argmax(pred_positive_all)+1
            elif im_pred == 1:
                intensity_score = 1
            elif im_pred == 2:
                intensity_score = 2
            elif im_pred == 3:
                intensity_score = 3

            allred_score = proportion_score + intensity_score
            print('Proportion Score: ',proportion_score)
            print('Intensity Score: ',intensity_score)            
            print('Allred Score: ',allred_score)
            # shutil.rmtree(roi_path, ignore_errors=True)
                    
            end_time=time.time()
            print("Execution time: ",end_time-start_time)
            print("Prediction time: ",end_prediction_time-start_prediction_time)
            print("Scoring time: ",end_time-start_scoring_time)

            f.write("\n")
            f.write("Image ID;Class Prediction;Class 0 (Negative);Class 1 (Weak);Class 2 (Moderate);Class 3 (Strong);Total Prediction;Total Positive;Class Positive Max;Positive Percentage;Proportion Score;Intensity Score;Allred Score;Execution Time;Prediction Time;Scoring Time \n")
            f.write("{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(id_image,im_pred,pred_c0,pred_c1,pred_c2,pred_c3,pred_total,pred_positive,np.argmax(pred_positive_all),pred_positive_100,proportion_score,intensity_score,allred_score,end_time-start_time,end_prediction_time-start_prediction_time,end_time-start_scoring_time))
            
        f.close()
        
        job.update(status=Job.RUNNING, progress=99, statusComment="Summarizing results...")
        job_data = JobData(job.id, "Generated File", "classification_results.csv").save()
        job_data.upload(output_path)

    finally:
        logging.info("Deleting folder %s", working_path)
        shutil.rmtree(working_path, ignore_errors=True)
        logging.debug("Leaving run()")


    job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.") 

if __name__ == "__main__":
    logging.debug("Command: %s", sys.argv)

    with cytomine.CytomineJob.from_cli(sys.argv) as cyto_job:
        run(cyto_job, cyto_job.parameters)

