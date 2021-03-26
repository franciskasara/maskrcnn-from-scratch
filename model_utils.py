import tensorflow as tf
import numpy as np
import os
import json
import nrrd
import matplotlib.pyplot as plt
import random
import time
import utils


#NEURALNETWORKS

def backboneNN():
    input_=tf.keras.layers.Input((512,512,1))

    x=tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(input_)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPooling2D((2, 2))(x)
    #256

    x=tf.keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same')(x)
    #x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPooling2D((2, 2))(x)
    #128


    x=tf.keras.layers.Conv2D(256, (3, 3), activation='relu',padding='same')(x)
    #x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPooling2D((2, 2))(x)
    #64

    x=tf.keras.layers.Conv2D(256, (5, 5), activation='relu',padding='same')(x)
    #x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPooling2D((2, 2))(x)
    #32

    featuremap=tf.keras.layers.Conv2D(256, (3, 3), activation='relu',padding='same',name='featuremap')(x)
    #x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPooling2D((2, 2))(featuremap)
    #16


    x=tf.keras.layers.Flatten()(x)
    output=tf.keras.layers.Dense(2,activation='sigmoid')(x) # in case of multiclass one-hot encoding we need a sigmoid at the end
    featuremapmodel=tf.keras.Model(input_,featuremap,name="CNN_fm")
    classifiermodel=tf.keras.Model(input_,output,name="CNN")

    featuremapmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.BinaryCrossentropy())
    classifiermodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.5))

    return classifiermodel,featuremapmodel



def rpnN(featuremap):
    #RPN modell

    initializer = tf.keras.initializers.GlorotNormal(seed=None)
    input_= tf.keras.layers.Input(shape=[None, None, featuremap.shape[-1]], name="rpn_INPUT")

    shared = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', strides=1, name='rpn_conv_shared',kernel_initializer=initializer)(input_)
    x = tf.keras.layers.Conv2D(5*2 , (1, 1), padding='valid', activation='linear',name='rpn_class_raw',kernel_initializer=initializer)(shared) 

    rpn_class_logits = tf.keras.layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)
    rpn_probs = tf.keras.layers.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits) # --> BG/FG

    # Bounding box refinement. [batch, H, W, depth]
    x = tf.keras.layers.Conv2D(5*4, (1, 1), padding="valid", activation='linear', name='rpn_bbox_pred',kernel_initializer=initializer)(shared) 

    # Reshape to [batch, anchors, 4]
    rpn_bbox = tf.keras.layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)
    outputs = [rpn_class_logits, rpn_probs, rpn_bbox]
    rpnN = tf.keras.models.Model(input_, outputs, name="RPN")

    return rpnN


def classheadNN(featurefilters,proposalcount,roisize):
    input_=tf.keras.layers.Input((proposalcount,roisize[0],roisize[1],featurefilters))

    x=tf.keras.layers.Conv2D(kernel_size=(1,1),padding='valid',activation='relu',filters=featurefilters)(input_)
    x=tf.debugging.check_numerics(x, 'x has nan')
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.ReLU()(x)

    x=tf.keras.layers.Conv2D(kernel_size=(1,1),padding='valid',activation='relu',filters=featurefilters)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.ReLU()(x)

    flatten=tf.keras.layers.Flatten()(x)

    beforeclass=tf.keras.layers.Dense(proposalcount*3,name='beforeclasspred')(flatten)#3: numofclasses + 1 for background
    beforeclass=tf.debugging.check_numerics(beforeclass, 'beforeclass has nan')

    class_logits = tf.keras.layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], proposalcount,3]),name='classpred')(beforeclass)
    class_probs= tf.keras.layers.Activation("softmax", name="classhead_class")(class_logits) # --> BG/FG



    beforebox=tf.keras.layers.Dense(3*4*proposalcount,activation='linear',name='beforeboxpred')(flatten) # 3 is the num of classes + 1 for background
    bboxpred = tf.keras.layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0],proposalcount, 3, 4]),name='boxrefinement')(beforebox) # for every roi for every class we predict dx,dy,dw,dh
    outputs=[class_logits,class_probs,bboxpred]
    classheadNN=tf.keras.Model(input_,outputs,name="classhead")
    
    return classheadNN

def maskheadNN(featurefilters,proposalcount,maskroisize):
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    input_=tf.keras.layers.Input((proposalcount,maskroisize[0],maskroisize[1],featurefilters))

    x=tf.keras.layers.Conv2D(kernel_size=(1,1),padding='same',activation='relu',filters=featurefilters/2)(input_)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.ReLU()(x)

    x=tf.keras.layers.Conv2D(kernel_size=(3,3),padding='same',activation='relu',filters=featurefilters/2)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.ReLU()(x)


    x=tf.keras.layers.Conv2D(kernel_size=(3,3),padding='same',activation='relu',filters=featurefilters/2)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.ReLU()(x)

    x=tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D(size=(2,2)))(x)
    x=tf.keras.layers.ReLU()(x)
    pred_mask=tf.keras.layers.Conv2D(kernel_size=(1,1),padding='same',activation='sigmoid',filters=2)(x) #2 filters, as we predict a mask for each class 

    maskheadNN=tf.keras.Model(input_,pred_mask,name="maskhead")
    
    return maskheadNN




#GENERATORS

def batchgenerator_cnn(datafolder,jsonfile,batchlen,numofdatas=None,fromimage=0):
    if numofdatas is None:
        numofdatas=len(os.listdir(datafolder))
    while True:
        indices = np.arange(fromimage,numofdatas)
        np.random.shuffle(indices)
        for batchstart in range(0+fromimage,fromimage+numofdatas,batchlen):
            x_batch=np.zeros((batchlen,512,512),dtype=np.float32)      
            y_batch=np.zeros((batchlen,2),dtype=np.float32)      
            filenames=[]
            for num,i in enumerate(indices[batchstart:batchstart+batchlen]):
                filename=str(i).zfill(6)+'.nrrd'
                data, h_=nrrd.read(os.path.join(datafolder,filename))
                x_batch[num]=data
                y_batch[num]=jsonfile[filename]['label']
                filenames.append(filename)
            if len(filenames)!=batchlen:
                continue
            else:
                x_batch=np.expand_dims(x_batch,-1)
                yield x_batch, y_batch
            
def batchgenerator_maskrcnn(datafolder,maskfolder,jsonfile,batchlen,numofdatas=None,mode='Complextrain',fromimage=0):
    if numofdatas is None:
        numofdatas=len(os.listdir(datafolder))
    indices = np.arange(fromimage,numofdatas)
    np.random.shuffle(indices)
    for batchstart in range(0+fromimage,fromimage+numofdatas,batchlen):
        x_batch=np.zeros((batchlen,512,512),dtype=np.float32)  
        y_batch=np.zeros((batchlen,2),dtype=np.float32)        
        m_batch=np.zeros((batchlen,512,512),dtype=np.float32)      
        bb_batch=[]
        filenames=[]
        for num,i in enumerate(indices[batchstart:batchstart+batchlen]):
            filename=str(i).zfill(6)+'.nrrd'
            data, h_=nrrd.read(os.path.join(datafolder,filename))
            x_batch[num]=data
            y_batch[num]=jsonfile[filename]['label']
            bbox=jsonfile[filename]['bbox']
            bb_batch.append(np.asarray(bbox))
            filenames.append(filename)
            if mode == 'Masktrain' or mode== 'Complextrain':
                mask, h_=nrrd.read(os.path.join(maskfolder,filename))
                m_batch[num]=mask           
        if (np.sum(np.sum(np.sum(bb_batch)))==0): # we can't have a batch full of only BG images, cause the boxloss only takes FGs --> it would go to Nan in case of a full BG batch
            continue
        elif len(filenames)!=batchlen:
            continue
        else:
            x_batch=np.expand_dims(x_batch,-1)
            if mode=='RPNtrain':
                yield x_batch, bb_batch
            elif mode=='Headtrain':
                yield x_batch,y_batch,bb_batch
            elif mode=='Masktrain':
                m_batch=np.expand_dims(m_batch,-1)
                yield x_batch,y_batch,bb_batch, m_batch, filenames
            else:
                m_batch=np.expand_dims(m_batch,-1)
                yield x_batch,y_batch,bb_batch, m_batch, filenames
                
#LOSSES               
                
def smooth_l1(y_true, y_pred):
    # Take absolute difference
    x = tf.abs(y_true - y_pred)
    # Find indices of values less than 1
    mask = tf.cast(tf.less(x, 1.0), "float32")
    # Loss calculation for smooth l1
    loss = (mask * (0.5 * x ** 2)) + (1 - mask) * (x - 0.5)
    return loss


def rpn_loss(rpn_logits,rpn_deltas, gt_labels,gt_deltas , indices, batchlen):
    
    '''
    rpn_logits,rpn_deltas: the predicted logits/deltas to all the anchors
    gt_labels,gt_deltas: the correct labels and deltas to the chosen training anchors
    indices: the indices of the chosen training anchors
    '''

    predicted_classes = tf.gather_nd(rpn_logits, indices)
    foregroundindices=indices[gt_labels.astype('bool')] #labels: 0:BG  1:FG
    predicted_deltas=tf.cast(tf.gather_nd(rpn_deltas, foregroundindices),tf.float32) #only the foreground anchors contribute to the box loss
    gt_deltas=tf.cast(tf.gather_nd(gt_deltas, foregroundindices),tf.float32)

    # Cross entropy loss

    lf=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    classloss = lf(gt_labels,predicted_classes)
    classloss=tf.reduce_mean(classloss)
    
    deltaloss=smooth_l1(gt_deltas,predicted_deltas)
    deltaloss=tf.reduce_mean(deltaloss)
    
    return classloss,deltaloss

def ch_loss(ch_logits, ch_deltas, gt_labels, gt_deltas, indices):

    predicted_classes = tf.gather_nd(ch_logits, indices)
    foregroundindices=tf.where(gt_labels<2)#where the label is 2 it means it is a backrground box -> we filter these out   

    classlabels=tf.cast(tf.gather_nd(gt_labels,foregroundindices),tf.int32) #the boxrefinement predicted deltas to each class; we need the proper classes to filter out the proper deltas
    idxs=tf.range(classlabels.shape)
    classlabels=tf.stack((idxs,classlabels),axis=1)# pred_deltas shape: (proposalcount,classes,4) -> we want to filter from axis1 -> easiest is to have 0 dimension ready with range
    
    pd=tf.gather_nd(ch_deltas,indices)# filter out proposed box deltas from all deltas
    gd=tf.gather_nd(gt_deltas,indices)# filter out proposed box deltas from all deltas
    
    predicted_deltas=tf.cast(tf.gather_nd(pd, foregroundindices),tf.float32) #only positive boxes contribute to the box refinement loss
    predicted_deltas=tf.cast(tf.gather_nd(predicted_deltas, classlabels),tf.float32)#only the proper classes box contribute to the box refinement loss
    gtdeltas=tf.cast(tf.gather_nd(gd, foregroundindices),tf.float32)
    

    not_empty_images=tf.where(tf.math.logical_not(tf.reduce_all(tf.equal(gt_labels,2),axis=-1)))
    not_empty_gts=tf.gather(gt_labels,not_empty_images,axis=0) #we dont want only background images in the classloss, that would inbalance the dataset
    not_empty_preds=tf.gather(predicted_classes,not_empty_images,axis=0)
    
    # Cross entropy loss
    lf=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    classloss = lf(not_empty_gts,not_empty_preds)
    classloss=tf.reduce_sum(classloss)
    
    deltaloss=smooth_l1(gtdeltas,predicted_deltas)
    deltaloss=tf.reduce_sum(deltaloss)

    
    return classloss,deltaloss

def mask_loss(pred_mask,gt_mask,gt_labels,indices):
    
    foregroundindices=tf.where(gt_labels<2)
    classlabels=tf.cast(tf.gather_nd(gt_labels,foregroundindices),tf.int32)
    idxs=tf.range(classlabels.shape)
    classlabels=tf.stack((idxs,classlabels),axis=1)
    
    pm=tf.cast(tf.gather_nd(pred_mask,indices),tf.float32)
    pm=tf.gather_nd(pm,foregroundindices)
    pm=tf.transpose(pm,[0,3,1,2])
    predicted_masks=tf.gather_nd(pm, classlabels)
    predicted_masks=tf.expand_dims(predicted_masks, axis=-1)

    
    gm=tf.cast(tf.gather_nd(gt_mask,indices),tf.float32)
    gt_masks=tf.cast(tf.gather_nd(gm,foregroundindices),tf.float32)
    gt_max=tf.reduce_max(gt_masks)
    gt_masks=tf.math.divide(gt_masks,gt_max) #we need a binary mask

    lf=tf.keras.losses.binary_crossentropy(gt_masks,predicted_masks)
    loss=tf.reduce_sum(lf)
    return loss
    
#TRAINSTEPS

def create_rpn_trainstep(rpnmodel,fmmodel):
    def rpn_trainstep(images, gt_box,allanchors,proposalcount,batchlen, rpn_optimizer):
        with tf.GradientTape() as gradientT:
            featuremaps=fmmodel(images)
            logits,probs,deltas = rpnmodel(featuremaps)    #Here we get the RPNn outputs
            indices,gt_deltas,gt_labels=utils.indices_deltas_labels(gt_box,allanchors,batchlen,proposalcount,mode='pixelwise')
            rpn_loss_class,rpn_loss_delta = rpn_loss(logits, deltas, gt_labels, gt_deltas, indices, batchlen)                
            rpn_loss_w=rpn_loss_class+rpn_loss_delta
        gradients_of_rpn = gradientT.gradient(rpn_loss_w, rpnmodel.trainable_variables)
        rpn_optimizer.apply_gradients(zip(gradients_of_rpn, rpnmodel.trainable_variables))
        return rpn_loss_w,rpn_loss_class,rpn_loss_delta
    return rpn_trainstep


def create_ch_trainstep(classheadmodel,rpnmodel,fmmodel):
    def ch_trainstep(images,labels, gt_box, allanchors,roisize,batchlen, ch_optimizer):
        with tf.GradientTape() as gradientT:
            featuremaps=fmmodel(images)
            rpn_logits,rpn_probs,rpn_boxes=rpnmodel(featuremaps)
            proposals=utils.get_proposals(rpn_probs,rpn_boxes,allanchors)            
            aligned_rois=utils.roi_align(featuremaps,proposals,roisize)
            ch_logits,ch_probs,ch_deltas= classheadmodel(aligned_rois)  
            indices,gt_deltas,gt_labels=utils.head_indices_deltas_labels(gt_box, labels, proposals, batchlen=batchlen, train_set_size=6)
            ch_classloss,ch_deltaloss = ch_loss(ch_logits, ch_deltas, gt_labels, gt_deltas, indices)   
            ch_loss_w=ch_classloss+ch_deltaloss
        gradients_of_classhead = gradientT.gradient(ch_loss_w, classheadmodel.trainable_variables)
        ch_optimizer.apply_gradients(zip(gradients_of_classhead, classheadmodel.trainable_variables))
        return ch_classloss,ch_deltaloss,
    return ch_trainstep

def create_mask_trainstep(maskheadmodel,rpnmodel,fmmodel):
    def mask_trainstep(images,labels, gt_box, masks,allanchors,maskroisize, batchlen, mask_optimizer):
        with tf.GradientTape() as gradientT:
            featuremaps=fmmodel(images)
            rpn_logits,rpn_probs,rpn_boxes=rpnmodel(featuremaps)
            proposals=utils.get_proposals(rpn_probs,rpn_boxes,allanchors)            
            indices,gt_deltas,gt_labels=utils.head_indices_deltas_labels(gt_box, labels, proposals, batchlen=batchlen, train_set_size=6)
            fm_rois,mask_rois=utils.mask_roi_align(featuremaps,masks,proposals,maskroisize)
            predicted_masks=maskheadmodel(fm_rois)
            maskloss=mask_loss(predicted_masks,mask_rois,gt_labels,indices)
        gradients_of_maskhead = gradientT.gradient(maskloss, maskheadmodel.trainable_variables)
        mask_optimizer.apply_gradients(zip(gradients_of_maskhead, maskheadmodel.trainable_variables))
        return maskloss
    return mask_trainstep


def create_complex_trainstep(fmmodel,rpnmodel,classheadmodel,maskheadmodel):
    def complex_trainstep(images,gt_labels, gt_box,masks,allanchors, batchlen,proposalcount,roisize,maskroisize, complex_optimizer):
        with tf.GradientTape() as gradientT:
            
            featuremaps=fmmodel(images)
            #RPN
            rpn_logits,rpn_probs,rpn_deltas = rpnmodel(featuremaps)    #Itt kapjuk meg az RPN kimeneteket
            rpn_indices,rpn_gt_deltas,rpn_gt_labels=utils.indices_deltas_labels(gt_box,allanchors,batchlen,proposalcount,mode='pixelwise')
            rpn_loss_class,rpn_loss_delta = rpn_loss(rpn_logits, rpn_deltas, rpn_gt_labels, rpn_gt_deltas, rpn_indices, batchlen)                
            rpn_loss_w=rpn_loss_class+rpn_loss_delta
            
            #CH-BoxRefinement
            proposals=utils.get_proposals(rpn_probs,rpn_deltas,allanchors)            
            ch_indices,ch_gt_deltas,ch_gt_labels=utils.head_indices_deltas_labels(gt_box, gt_labels, proposals, batchlen=batchlen, train_set_size=6)
            aligned_rois=utils.roi_align(featuremaps,proposals,roisize)
            ch_logits,ch_probs,ch_deltas= classheadmodel(aligned_rois)  
            ch_classloss,ch_deltaloss = ch_loss(ch_logits, ch_deltas, ch_gt_labels, ch_gt_deltas, ch_indices)   
            ch_loss_w=ch_classloss+ch_deltaloss
            
            #Maskhead
            fm_rois,mask_rois=utils.mask_roi_align(featuremaps,masks,proposals,maskroisize)
            predicted_masks=maskheadmodel(fm_rois)
            maskloss=mask_loss(predicted_masks,mask_rois,ch_gt_labels,ch_indices)
            
            #Complex loss and variables
            complexloss=rpn_loss_w+ch_loss_w+maskloss
            complextrainables=fmmodel.trainable_variables+rpnmodel.trainable_variables+classheadmodel.trainable_variables+maskheadmodel.trainable_variables
        
        gradients = gradientT.gradient(complexloss, complextrainables)
        complex_optimizer.apply_gradients(zip(gradients, complextrainables))
        return rpn_loss_w,ch_loss_w,maskloss
    return complex_trainstep



#TRAINLOOPS     
    

def train_rpn(rpnmodell, fmmodel,allanchors,proposalcount,datafolder,maskfolder,jsonfile,batchlen, epochs,trainfromimage=0,numofdatas=None,rpn_optimizer=None):
    if numofdatas is None:
        numofdatas=len(os.listdir(datafolder))
    if rpn_optimizer is None:
        rpn_optimizer = tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9,clipnorm=5.0)
    trainstep=create_rpn_trainstep(rpnmodell,fmmodel)
    for epoch in range(epochs):
        start = time.time()
        losses=[]
        losses_c=[]
        losses_d=[]
        batchgen=batchgenerator_maskrcnn(datafolder,maskfolder,jsonfile,batchlen,numofdatas=numofdatas,mode='RPNtrain',fromimage=trainfromimage)
        for num,image_batch in enumerate(batchgen):
            x,bb=image_batch
            l,lc,ld=trainstep(x,bb,allanchors,proposalcount,batchlen,rpn_optimizer)
            losses.append(l)
            losses_c.append(lc)
            losses_d.append(ld)

        end = time.time()
        print(round(end-start),'sec. \t',epoch,'.epoch:\t loss(sum,c,bb):\t', np.mean(losses),np.mean(losses_c),np.mean(losses_d))
        

        
def train_classhead(classheadmodel,rpnmodel, fmmodel,allanchors,roisize,numofdatas,datafolder,maskfolder,jsonfile,batchlen, epochs,trainfromimage=0,ch_optimizer=None):
    if ch_optimizer is None:
        #ch_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001,clipvalue=0.5,clipnorm=1)
        ch_optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001,momentum=0.9,clipnorm=5.0)
    trainstep=create_ch_trainstep(classheadmodel,rpnmodel,fmmodel)
    for epoch in range(epochs):
        losses_c=[]
        losses_d=[]
        start = time.time()
        batchgen=batchgenerator_maskrcnn(datafolder,maskfolder,jsonfile,batchlen,numofdatas,mode='Headtrain',fromimage=trainfromimage)
        for num,image_batch in enumerate(batchgen):
            x,y,bb=image_batch
            classloss,deltaloss=trainstep(x,y,bb,allanchors,roisize,batchlen,ch_optimizer)
            losses_c.append(classloss)
            losses_d.append(deltaloss)
        end = time.time()
        print(round(end-start),'sec. \t',epoch,'.epoch: \t loss(c,bb):\t',np.mean(losses_c),np.mean(losses_d))
        
        
def train_maskhead(maskheadmodel,rpnmodel, fmmodel,allanchors,maskroisize,numofdatas,datafolder,maskfolder,jsonfile,batchlen, epochs,trainfromimage=0,mask_optimizer=None):
    if mask_optimizer is None:
        mask_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001,clipvalue=0.5,clipnorm=1)
        #mask_optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001,momentum=0.9,clipnorm=5.0)
    trainstep=create_mask_trainstep(maskheadmodel,rpnmodel,fmmodel)
    for epoch in range(epochs):
        losses_m=[]
        start = time.time()
        batchgen=batchgenerator_maskrcnn(datafolder,maskfolder,jsonfile,batchlen,numofdatas,mode='Masktrain',fromimage=trainfromimage)
        for num,image_batch in enumerate(batchgen):
            x,y,bb,m,fnames=image_batch
            maskloss=trainstep(x,y,bb,m,allanchors,maskroisize,batchlen,mask_optimizer)
            losses_m.append(maskloss)
        end = time.time()
        print(round(end-start),'sec. \t',epoch,'.epoch: \t loss:\t',np.mean(losses_m))
        

def train_complex(fmmodel,rpnmodel,classheadmodel,maskheadmodel,allanchors,proposalcount,roisize,maskroisize,numofdatas,datafolder,maskfolder,jsonfile,batchlen, epochs,trainfromimage=0,complex_optimizer=None):
    if complex_optimizer is None:
        complex_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001,clipvalue=0.5,clipnorm=1)
        #mask_optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001,momentum=0.9,clipnorm=5.0)
    trainstep=create_complex_trainstep(fmmodel,rpnmodel,classheadmodel,maskheadmodel)
    for epoch in range(epochs):
        rpnlosses=[]
        chlosses=[]
        masklosses=[]
        
        start = time.time()
        batchgen=batchgenerator_maskrcnn(datafolder,maskfolder,jsonfile,batchlen,numofdatas,mode='Masktrain',fromimage=trainfromimage)
        for num,image_batch in enumerate(batchgen):
            x,y,bb,m,fnames=image_batch
            rpnloss,chloss,maskloss=trainstep(x,y,bb,m,allanchors,batchlen,proposalcount,roisize,maskroisize,complex_optimizer)
            
            rpnlosses.append(rpnloss)
            chlosses.append(chloss)
            masklosses.append(maskloss)

        end = time.time()
        print(round(end-start),'sec. \t',epoch,'.epoch: \t rpn loss:',np.mean(rpnlosses),'\tch loss:',np.mean(chlosses),'\tmask loss:',np.mean(masklosses))
        
        


#PREDICTS       
        
def predict_rpn(rpnmodel,x_batch,fmmodel,returnFM=False):
    featuremaps=fmmodel(x_batch)
    rpn_logits,rpn_probs,rpn_bbox = rpnmodel(featuremaps)    #Itt kapjuk meg az RPN kimeneteket
    if returnFM:
        return rpn_probs,rpn_bbox,featuremaps
    else:
        return rpn_probs,rpn_bbox

def predict_ch(image_batch,anchors,batchlen,fmmodel,rpnmodel,classheadmodel,roisize=[5,5]):
    
    
    featuremaps=fmmodel(image_batch)
    rpn_logits,rpn_probs,rpn_boxes=rpnmodel(featuremaps)
    proposals=utils.get_proposals(rpn_probs,rpn_boxes,anchors)            
    rois=utils.roi_align(featuremaps,proposals,roisize)
    chlogits,chprobs,chdeltas= classheadmodel(rois)  

    
    #we need the proposals witch are predicted to be foreground, than we need the exact classpredictions
    classes=np.argmax(chprobs,axis=-1)
    foregroundindices=tf.where(classes<2)
    classlabels=tf.cast(tf.gather_nd(classes,foregroundindices),tf.int32)
    
    idxs=tf.range(classlabels.shape)
    classlabelindices=tf.stack((idxs,classlabels),axis=1)   
    
    softmax_scores=tf.cast(tf.gather_nd(chprobs,foregroundindices),tf.float32)
    scores=np.amax(softmax_scores,axis=-1)
  
    
    # we need to get the proper refinements; the foregrund refinements, and the channel of the proper refinement
    pred_box=tf.gather_nd(chdeltas,foregroundindices)
    pred_box=tf.cast(tf.gather_nd(pred_box,classlabelindices),tf.float64)
    
    #maybe the classhead predicted a class to a proposal, but if that proposal is empty
    #so the RPN said it is background, but we padded it to match the classhead modell input shape, we need to filter that out too
    
    fg_proposals=tf.gather_nd(proposals,foregroundindices)
    
    fgproposalindices=[]
    for num,proposal in enumerate(fg_proposals):
        if np.all(np.not_equal(proposal,0)):
            fgproposalindices.append(num)
    fgproposalindices=tf.cast(fgproposalindices,tf.int32)
    
    foregroundindices=tf.gather(foregroundindices,fgproposalindices)   
    fg_proposals=tf.gather(fg_proposals,fgproposalindices)             
    classlabels=tf.gather(classlabels,fgproposalindices)
    softmax_scores=tf.gather(softmax_scores,fgproposalindices)       
    pred_box=tf.gather(pred_box,fgproposalindices)                           

    preds_per_images=np.bincount(foregroundindices[:,0])

    #we want to shift the foreground proposals
    refined_boxes=tf.cast(utils.shift_bbox_pixelwise(fg_proposals,pred_box),tf.float32)
        
    if np.all(np.equal(preds_per_images,0)):
        print('I found nothing on the batch.',preds_per_images)
        return 0,0,0,0
    else:
        adjusted_boxes,adjusted_scores,adjusted_labels=utils.adjust_to_batch(fg_proposals,classlabels,softmax_scores,refined_boxes,preds_per_images,batchlen)    
        #return fg_proposals,pred_masks,classlabels,softmax_scores,refined_boxes,preds_per_images
        return adjusted_boxes,adjusted_scores,adjusted_labels


def predict_all(image_batch,anchors,batchlen,fmmodel,rpnmodel,classheadmodel,maskheadmodel,roisize=[5,5],maskroisize=[14,14]):
    
    
    featuremaps=fmmodel(image_batch)
    rpn_logits,rpn_probs,rpn_boxes=rpnmodel(featuremaps)
    proposals=utils.get_proposals(rpn_probs,rpn_boxes,anchors)            
    rois=utils.roi_align(featuremaps,proposals,roisize)
    fm_rois=utils.roi_align(featuremaps,proposals,maskroisize)
    predicted_masks=maskheadmodel(fm_rois)
    chlogits,chprobs,chdeltas= classheadmodel(rois)  

    
    #we need the proposals witch are predicted to be foreground, than we need the exact classpredictions
    classes=np.argmax(chprobs,axis=-1)
    foregroundindices=tf.where(classes<2)
    classlabels=tf.cast(tf.gather_nd(classes,foregroundindices),tf.int32)
    
    idxs=tf.range(classlabels.shape)
    classlabelindices=tf.stack((idxs,classlabels),axis=1)   
    
    softmax_scores=tf.cast(tf.gather_nd(chprobs,foregroundindices),tf.float32)
    scores=np.amax(softmax_scores,axis=-1)
    # we need to get the proper masks; the foregrund masks, and the channel of the proper class
    pred_masks=tf.gather_nd(predicted_masks,foregroundindices)
    
    
    pred_masks=tf.transpose(pred_masks,[0,3,1,2])
    pred_masks=tf.gather_nd(pred_masks, classlabelindices)
    pred_masks=tf.expand_dims(pred_masks, axis=-1)
    
    
    # we need to get the proper refinements; the foregrund refinements, and the channel of the proper refinement
    pred_box=tf.gather_nd(chdeltas,foregroundindices)
    pred_box=tf.cast(tf.gather_nd(pred_box,classlabelindices),tf.float64)
    
    #maybe the classhead predicted a class to a proposal, but if that proposal is empty
    #so the RPN said it is background, but we padded it to match the classhead modell input shape, we need to filter that out too
    
    fg_proposals=tf.gather_nd(proposals,foregroundindices)
    
    fgproposalindices=[]
    for num,proposal in enumerate(fg_proposals):
        if np.all(np.not_equal(proposal,0)):
            fgproposalindices.append(num)
    fgproposalindices=tf.cast(fgproposalindices,tf.int32)
    
    foregroundindices=tf.gather(foregroundindices,fgproposalindices)   
    fg_proposals=tf.gather(fg_proposals,fgproposalindices)             
    classlabels=tf.gather(classlabels,fgproposalindices)
    softmax_scores=tf.gather(softmax_scores,fgproposalindices)       
    pred_box=tf.gather(pred_box,fgproposalindices)                           
    pred_masks=tf.gather(pred_masks,fgproposalindices)        
    
    
    preds_per_images=np.bincount(foregroundindices[:,0],minlength=batchlen)
    
    
    #we want to shift the foreground proposals
    refined_boxes=tf.cast(utils.shift_bbox_pixelwise(fg_proposals,pred_box),tf.float32)
        
    if np.all(np.equal(preds_per_images,0)):
        print('I found nothing on the batch.',preds_per_images)
        return 0,0,0,0
    else:
        adjusted_masks,adjusted_boxes,adjusted_scores,adjusted_labels=utils.adjust_to_batch(fg_proposals,classlabels,softmax_scores,refined_boxes,preds_per_images,batchlen,pred_masks)    
        #return fg_proposals,pred_masks,classlabels,softmax_scores,refined_boxes,preds_per_images
        return adjusted_masks,adjusted_boxes,adjusted_scores,adjusted_labels
