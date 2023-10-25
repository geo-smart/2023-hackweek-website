'''
This script includes all packages and functions required by the book chapter
'''
# evaluate random forest algorithm for classification
from numpy import arange
from sklearn.datasets import make_classification

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from rasterio.plot import show
from sklearn.model_selection import train_test_split
import seaborn as sn
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import joblib
import rasterio
import pandas as pd
import matplotlib

# clip the ASO data to planet extent
import fiona
import rasterio.mask


def calculate_metrics(df):
    
    # true positive and true negative
    subdf = df[df.predict == df.obs]
    TP = len(subdf[subdf.predict == 1].index)
    TN = len(subdf[subdf.predict == 0].index)

    # false positive and false negative
    subdf = df[df.predict != df.obs]
    FN = len(subdf[subdf.predict == 0].index)
    FP = len(subdf[subdf.predict == 1].index)

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    balanced_accuracy = (sensitivity+specificity)/2
    accuracy = (TP+TN)/(TP+TN+FP+FN)

    f1 =  0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    out = pd.DataFrame(data = {'precision': [precision], 
                               'recall':[recall], 
                               'f1':[f1],
                               'balanced_accuracy':[balanced_accuracy], 
                               'accuracy':[accuracy]})
    return out

# the function used to predict binary snow cover
def run_sca_prediction(dir_raster, dir_out, nodata_flag, model):
    """
    This function predicts binary snow cover for planet satellite images using 
    the pre-trained random forest model 
    
    :param dir_raster: the directory or the file of planet images
    :param dir_out: the directory where output snow cover images will be stored
    :param nodata_flag: the value used to represent no data in the predicted snow cover image
    defult value is 9.
    model: the model used to predict snow cover
    
    """
    # if output directory not exist then creat the output directory
    if not os.path.exists(dir_out): os.mkdir(dir_out)
    
    # if dir_raster is a directory, then find all images with 'SR' flag, meaning surface reflectance data
    if os.path.isdir(dir_raster):
        file_list = glob.glob(dir_raster + './**/*SR*.tif', recursive = True)
    elif os.path.isfile(dir_raster):
        file_list = [dir_raster]
        
    for f in file_list:
        print('Start to predict:'.format(), os.path.basename(f))

        with rasterio.open(f, 'r') as ds:
            arr = ds.read()  # read all raster values

        print("Image dimension:".format(), arr.shape)  # 
        X_img = pd.DataFrame(arr.reshape([4,-1]).T)
        X_img.columns = ['blue','green','red','nir']
        X_img['nodata_flag'] = np.where(X_img['blue']==0, -1, 1)
        
        X_img = X_img/10000 # scale surface reflectance to 0-1
        # run model prediction
        y_img = model.predict(X_img.iloc[:,0:4])
        
        out_img = pd.DataFrame()
        out_img['label'] = y_img
        out_img['nodata_flag'] = X_img['nodata_flag']
        out_img['label'] = np.where(out_img['nodata_flag'] == -1, nodata_flag, out_img['label'])
        # Reshape our classification map
        img_prediction = out_img['label'].to_numpy().reshape(arr[0,:, :].shape)

        
        file_out = dir_out + os.path.basename(f)[0:-4] + '_SCA.tif'
        print("Save SCA map to: ".format(),file_out)
        with rasterio.open(
                        file_out, "w",
                        driver = "GTiff",
                        transform = ds.transform,
                        dtype = rasterio.uint8,
                        count = 1,
                        crs = ds.crs,
                        width = ds.width,
                        height = ds.height) as dst:
                    dst.write(img_prediction, indexes = 1)
                
                
# model evaluation CA
def SCA_model_evaluation(dir_img_ext, dir_sca, dir_aso,flag_mask, dir_watermask, dir_glaciermask):
    """
    This function evaluate snow cover mapping accuracy 
    
    :param dir_img_ext: the ESRI shapefile of the image extent
    :param dir_sca: the predicted SCA 
    :param dir_aso: the ASO snow depth
    :param dir_watermask: water mask for Tuolumne CA; Only used for CA
    :param dir_glaciermask: glacier mask for Tuolumne CA; Only used in CA
    
    """
    print('Start SCA Evaluation: --------')


    with fiona.open(dir_img_ext, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rasterio.open(dir_aso) as src:
        r_aso = rasterio.mask.mask(src, shapes, crop=True)

    with rasterio.open(dir_sca) as src:
        r_predict = rasterio.mask.mask(src, shapes, crop=True)


    if flag_mask:
        with rasterio.open(dir_watermask) as src:
            r_watermask = rasterio.mask.mask(src, shapes, crop=True)

        with rasterio.open(dir_glaciermask) as src:
            r_glaciermask = rasterio.mask.mask(src, shapes, crop=True)

    df = pd.DataFrame()
    df['predict'] = r_predict[0].ravel()
    df['obs'] = r_aso[0].ravel()
    # get binary snow cover from ASO snow depth
    df.obs = np.where(df.obs > 0.1, 1, 0)

    if flag_mask:
        df['watermask'] = r_watermask[0].ravel()
        df['glaciermask'] = r_watermask[0].ravel()
        # remove NA 
        df = df[(df.predict >= 0) & (df.watermask != 0) & (df.glaciermask != 0)]
    else:
        df = df[(df.predict >= 0)]
    

    print("overall model performance:")
    print(calculate_metrics(df))
    
    
def evaluate_model(model, X, y):
    """
    Test model sensitivity 
    This function evaluates a given model using k-fold cross-validation
    """
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=100, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

# get a list of models to evaluate
def get_models_size():
    models = dict()
    for i in np.concatenate((arange(0.01, 0.1, 0.01), arange(0.1, 1.0, 0.1))):
        key = '%.2f' % i
        if i == 1.0:
            i = None
        models[key] = RandomForestClassifier(max_samples=i)
    return models

def get_models_feature():
    models = dict()
    for i in range(1,5):
        models[str(i)] = RandomForestClassifier(max_features=i)
    return models

def get_models_tree():
    models = dict()
    n_trees = [1,2,3,4,5,10,20,50,100,200,800,1000]
    for n in n_trees:
        models[str(n)] = RandomForestClassifier(n_estimators=n)
    return models

def get_models_depth():
    models = dict()
    depths = [i for i in range(1,20)] + [None]
    for n in depths:
        models[str(n)] = RandomForestClassifier(max_depth=n)
    return models

