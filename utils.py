
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
import random
import nrrd
import os



def generate_anchors(featuremap,orig_shape=512,anchor_sizes=[39,46,52,58,65], anchor_ratios=[1], anchor_stride=1):
    feature_shapes=featuremap.shape[2]
    feature_strides=orig_shape/featuremap.shape[2]
    anchors = []

    # All combinations of indices
    x = np.arange(0, feature_shapes, anchor_stride) * feature_strides #[  0  16  32  48  64 ... 480 496] 
    y = np.arange(0, feature_shapes, anchor_stride) * feature_strides
    x, y = np.meshgrid(x, y)  #shapes: 32x32

    # All combinations of indices, and shapes
    width, x = np.meshgrid(anchor_sizes, x)
    height, y = np.meshgrid(anchor_sizes, y)

    # Reshape indices and shapes
    x = x.reshape((-1, 1))  
    y = y.reshape((-1, 1))
    width = width.flatten().reshape((-1, 1))
    height = height.flatten().reshape((-1, 1))

    # Create the centers coordinates and shapes for the anchors
    bbox_centers = np.concatenate((y, x), axis=1)
    bbox_shapes = np.concatenate((height, width), axis=1)

    # Restructure as [y1, x1, y2, x2]
    bboxes = np.concatenate((bbox_centers - bbox_shapes / 2, bbox_centers + bbox_shapes / 2), axis=1)

    # Anchors are created for each feature map
    anchors.append(bboxes)
    print('Num of generated anchors:\t',len(bboxes))
    
    anchors=np.concatenate(anchors, axis=0)
    anchors=anchors
    return anchors

def calculate_ious(bbox,anchors):
    
    anchorarea = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1]) # area = width * height
    bboxarea= (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    y1 = np.maximum(bbox[0], anchors[:, 0]) 
    y2 = np.minimum(bbox[2], anchors[:, 2]) 
    x1 = np.maximum(bbox[1], anchors[:, 1]) 
    x2 = np.minimum(bbox[3], anchors[:, 3]) 
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = bboxarea + anchorarea[:] - intersection[:]
    iou = intersection / union
    
    return iou


def calculate_pixelwise_deltas(bbox,anchors,numof):
    
    assert len(anchors.shape)==2, "2 dimenzios anchors shape kell. Kapott: "+str(anchors.shape)
    assert len(bbox.shape)==1, "1 dimenzios bbox shape kell. Kapott: "+str(bbox.shape)
    
    deltas=np.zeros((numof,4)) #predicted dx,dy,dw,dh for each anchor
    
    anchor_widths=anchors[:, 2]-anchors[:, 0]
    anchor_heights=anchors[:, 3]-anchors[:, 1]
    anchor_centerx=anchors[:,0]+anchor_widths[:]/2
    anchor_centery=anchors[:,1]+anchor_heights[:]/2

    bbox_width =bbox[2] - bbox[0]
    bbox_height =bbox[3] - bbox[1]
    bbox_centerx=bbox[0]+bbox_width/2
    bbox_centery=bbox[1]+bbox_height/2
    
    dw=bbox_width-anchor_widths[:]
    dh=bbox_height-anchor_heights[:]
    dx=bbox_centerx-anchor_centerx[:]
    dy=bbox_centery-anchor_centery[:]

    for anchor in range(numof):
        deltas[anchor]=[dx[anchor],dy[anchor],dw[anchor],dh[anchor]]
    
    return deltas

def calculate_exponential_deltas(bbox,anchors,numof):
    raise NotImplementedError('Exponential anchorbox shifting is not implemented')


def indices_deltas_labels(batch_of_bboxes, anchors,batchlen, train_set_size=20,mode='pixelwise'):  
    
    num_of_anchors=len(anchors)
    batch_of_bboxes=np.array(batch_of_bboxes)
    
    batch_of_indices=np.zeros((batchlen,train_set_size,2),dtype=np.int32)
    batch_of_deltas=np.zeros((batchlen,num_of_anchors,4))
    batch_of_labels=np.zeros((batchlen,train_set_size))

    for im in range(batchlen):   
        bboxes_im=np.asarray(batch_of_bboxes[im])
        num_of_bboxes=bboxes_im.shape[0]
        
        indices=np.zeros((train_set_size,2),dtype=np.int32)
        deltas=np.zeros((num_of_anchors,4))
        boxlabels=np.zeros((train_set_size))
        
        if num_of_bboxes>1:
            bbox_ious=np.zeros((num_of_bboxes,num_of_anchors)) #Intersection over union score for each bbox-anchor pair
            bbox_deltas=np.zeros((num_of_bboxes,num_of_anchors,4)) #desired delta x,y,h,w for each bbox-anchor pair --> RPN shoult predict these
            ious=np.zeros((num_of_anchors))
            
            for bboxnum,bbox in enumerate(bboxes_im):
                if mode=='pixelwise':
                    bbox_deltas[bboxnum]=calculate_pixelwise_deltas(bbox,anchors,num_of_anchors)
                else:
                    bbox_deltas[bboxnum]=calculate_exponential_deltas(bbox,anchors,num_of_anchors)
                bbox_ious[bboxnum]=calculate_ious(bbox,anchors)

            #we want to train the anchors to move to the nearest bbox, if there are more --> so even if there are more bboxes, we only have one delta/iou value for each anchor
            for anchor in range(num_of_anchors):
                nearest_bbox=np.argmax(bbox_ious[:,anchor])
                deltas[anchor]=bbox_deltas[nearest_bbox,anchor]  
                ious[anchor]=bbox_ious[nearest_bbox,anchor]  
                    
        else:
            if np.all(np.equal(bboxes_im,0)): # if there are no masks on the image, the bbox of it is [0,0,0,0]
                sampledanchors=random.sample(range(0, num_of_anchors), train_set_size)
                indices=[[im,x] for x in sampledanchors]
                batch_of_indices[im]=indices
                batch_of_deltas[im]=deltas
                batch_of_labels[im]=boxlabels
                continue
                
            else:
                bbox=bboxes_im[0]
                if mode=='pixelwise':
                    deltas=calculate_pixelwise_deltas(bbox,anchors,num_of_anchors)
                else:
                    deltas=calculate_exponential_deltas(bbox,anchors,num_of_anchors)
                ious=calculate_ious(bbox,anchors)
                
        #we choose anchors with IoU>0.5 values to be foreground boxes, with IoU<0.1 to be backround boxes    
        num=0
        bg_indices=[]
        for anchor in range(num_of_anchors):
            if ious[anchor]>0.5:
                indices[num]=[im,anchor]
                if num<train_set_size//2:
                    num+=1
            elif ious[anchor]<0.1:
                bg_indices.append(anchor)

        # around half of the set consists of foreground boxes, half of it will be a randomly sampled set of background boxes
        sampledanchor=random.sample(bg_indices,train_set_size-num)
        indices[num:]=[[im,x] for x in sampledanchor]
        boxlabels[0:num]=1
        boxlabels[num:]=0


        batch_of_indices[im]=indices
        batch_of_deltas[im]=deltas
        batch_of_labels[im]=boxlabels
        
    return  batch_of_indices,batch_of_deltas, batch_of_labels



def head_indices_deltas_labels(batch_of_bboxes, batch_of_gt_labels, proposals ,batchlen, train_set_size=6,mode='pixelwise'):  
    
    num_of_proposals=proposals.shape[1]
    batch_of_bboxes=np.array(batch_of_bboxes)
    
    batch_of_indices=np.zeros((batchlen,train_set_size,2),dtype=np.int32)
    batch_of_deltas=np.zeros((batchlen,num_of_proposals,4))
    batch_of_labels=np.zeros((batchlen,train_set_size))

    for im in range(batchlen):   
        bboxes_im=np.asarray(batch_of_bboxes[im])
        num_of_bboxes=bboxes_im.shape[0]
        gt_label=batch_of_gt_labels[im] # [0,1], [1,0] when having one bounding box, [1,1] when having two, [0,0] when having 0.
        proposal_of_image=proposals[im]
        
        indices=np.zeros((train_set_size,2),dtype=np.int32)
        deltas=np.zeros((num_of_proposals,4))
        boxlabels=np.zeros((train_set_size))
        nearest_bboxes=np.zeros((num_of_proposals))

        if num_of_bboxes>1:
            bbox_ious=np.zeros((num_of_bboxes,num_of_proposals)) #Intersection over union score for each bbox-anchor pair
            bbox_deltas=np.zeros((num_of_bboxes,num_of_proposals,4)) #desired delta x,y,h,w for each bbox-anchor pair --> RPN shoult predict these
            ious=np.zeros((num_of_proposals))
            
            for bboxnum,bbox in enumerate(bboxes_im):
                if mode=='pixelwise':
                    bbox_deltas[bboxnum]=calculate_pixelwise_deltas(bbox,proposal_of_image,num_of_proposals)
                else:
                    bbox_deltas[bboxnum]=calculate_exponential_deltas(bbox,proposal_of_image,num_of_proposals)
                bbox_ious[bboxnum]=calculate_ious(bbox,proposal_of_image)

            #we want to train the anchors to move to the nearest bbox, if there are more --> so even if there are more bboxes, we only have one delta/iou value for each anchor
            for proposal in range(num_of_proposals):
                nearest_bbox=np.argmax(bbox_ious[:,proposal])
                nearest_bboxes[proposal]=nearest_bbox
                deltas[proposal]=bbox_deltas[nearest_bbox,proposal]  
                ious[proposal]=bbox_ious[nearest_bbox,proposal] 
                

                    
        else:
            if np.all(np.equal(bboxes_im,0)): # if there are no masks on the image, the bbox of it is [0,0,0,0]
                sampledanchors=random.sample(range(0, num_of_proposals), train_set_size)
                indices=[[im,x] for x in sampledanchors]
                boxlabels=boxlabels+2 # 2 (numofclasses+1) is the label of background
                batch_of_indices[im]=indices
                batch_of_deltas[im]=deltas
                batch_of_labels[im]=boxlabels
                continue
            

            else:
                bbox=bboxes_im[0]
                if mode=='pixelwise':
                    deltas=calculate_pixelwise_deltas(bbox,proposal_of_image,num_of_proposals)
                else:
                    deltas=calculate_exponential_deltas(bbox,proposal_of_image,num_of_proposals)
                ious=calculate_ious(bbox,proposal_of_image)
                nearest_bboxes=nearest_bboxes+np.argmax(gt_label)
                
        #we choose anchors with IoU>0.5 values to be foreground boxes, with 0.1<IoU<0.5 to be backround boxes    
        num=0
        bg_indices=[]
        for proposal in range(num_of_proposals):
            if ious[proposal]>0.4:
                indices[num]=[im,proposal]
                boxlabels[num]=nearest_bboxes[proposal]
                if num<train_set_size//2:
                    num+=1
            else:
                bg_indices.append(proposal)

        # around half of the set consists of foreground boxes, half of it will be a randomly sampled set of background boxes
        sampledanchor=random.sample(bg_indices,train_set_size-num)
        indices[num:]=[[im,x] for x in sampledanchor]
        boxlabels[num:]=2 # 2 (numofclasses+1) is the label of background


        batch_of_indices[im]=indices
        batch_of_deltas[im]=deltas
        batch_of_labels[im]=boxlabels

    return  batch_of_indices,batch_of_deltas, batch_of_labels


def read_batch(datafolder,maskfolder,jsonfile,batchlen=5,start=0):
    
    x_batch=np.zeros((batchlen,512,512))
    y_batch=np.zeros((batchlen,2))
    m_batch=np.zeros((batchlen,512,512))
    bb_batch=[]

    for num,imnum in enumerate(range(start,start+batchlen)):
        filename=str(imnum).zfill(6)+'.nrrd'
        im,h_=nrrd.read(os.path.join(datafolder,filename))
        mask,h_=nrrd.read(os.path.join(maskfolder,filename))
        x_batch[num]=im
        m_batch[num]=mask
        y_batch[num]=jsonfile[filename]['label']
        bbox=jsonfile[filename]['bbox'] 
        bb_batch.append(bbox)
        
    x_batch=np.expand_dims(x_batch,-1)
    m_batch=np.expand_dims(m_batch,-1)
    return x_batch,m_batch,bb_batch,y_batch



def draw_bbox(bboxparam):
    # Convert the bounding box to 4 lines in matplotlib to visualize it. boundingbox=[min_x,min_y,max_x,max_y]
    #in matplotlib line=start_x,end_x,start_y,end_y
    #so line by line: lowerline=[x1,x2],[y1,y1] #upperline=[x1,x2],[y2,y2] #leftsideline=[x1,x1],[y1,y2] #rightsideline=[x2,x2],[y1,y2]
    y1=bboxparam[0]
    y2=bboxparam[2]
    x1=bboxparam[1]
    x2=bboxparam[3]
    boxlines=[x1,x2],[y1,y1],[x1,x2],[y2,y2],[x1,x1],[y1,y2],[x2,x2],[y1,y2]
    return boxlines


def shift_bbox_pixelwise(anchors,predicted_deltas):
    
    assert len(anchors.shape)==2, "Anchor shape must be 2 dimensions. We got: "+str(anchors.shape)
    assert len(predicted_deltas.shape)==2, "predicted_deltas shape must be 2 dimensions. We got: "+str(predicted_deltas.shape)
    anchor_widths=anchors[:,2]-anchors[:, 0]
    anchor_heights=anchors[:,3]-anchors[:, 1]
    anchor_centerx=anchors[:,0]+anchor_widths[:]/2
    anchor_centery=anchors[:,1]+anchor_heights[:]/2

    pred_xc=anchor_centerx[:]+predicted_deltas[:,0]
    pred_yc=anchor_centery[:]+predicted_deltas[:,1]
    pred_widths=anchor_widths[:]+predicted_deltas[:,2]
    pred_heights=anchor_heights[:]+predicted_deltas[:,3]

    predx1=pred_xc[:]-pred_widths[:]/2
    predy1=pred_yc[:]-pred_heights[:]/2
    predx2=pred_xc[:]+pred_widths[:]/2
    predy2=pred_yc[:]+pred_heights[:]/2

    batch_of_boxes=np.stack([predx1, predy1, predx2, predy2], axis=1)
    return batch_of_boxes
            

def shift_bbox_exponential(anchors,predicted_deltas):
    raise NotImplementedError('Exponential anchorbox shifting is not implemented')
            
                


def visualize_rpn_result(image_batch,pred_scores,pred_deltas,anchors,proposal_count=20,mode='pixelwise'):

    proposals=get_proposals(pred_scores,pred_deltas,anchors,proposal_count)
    for num,image in enumerate(image_batch):
        plt.figure()
        plt.imshow(image,cmap='gray')
        for pred_bbox in proposals[num]:
            plt.plot(*draw_bbox(pred_bbox),color='red',linewidth=0.5, alpha=1)
    plt.show()
    
def visualize_ch_results(image_batch,predicted_label_batch,predicted_boxes_batch,predicted_scores_batch,classdict,batchlen):
    
    for i in range(batchlen):
        plt.figure()
        plt.imshow(image_batch[i],cmap='gray')
        for num,box in enumerate(predicted_boxes_batch[i]):
            if np.all(np.equal(box,0)):
                continue
            else:
                plt.plot(*draw_bbox(box),linewidth=2, alpha=1, color='pink')
                plt.text(box[1]+50,box[0]-5,predicted_scores_batch[i][num],color='pink',fontsize=12)
                plt.text(box[1],box[0]-5,classdict[predicted_label_batch[i][num]],color='pink',fontsize=12)

    
    
def visualize_results(image_batch,predicted_mask_batch,predicted_label_batch,predicted_boxes_batch,predicted_scores_batch,classdict,batchlen,gt_mask_batch=None):
    
    if batchlen>1:
        if gt_mask_batch is None:
            f, axarr = plt.subplots(batchlen,2)
        else:
            f, axarr = plt.subplots(batchlen,3)
        for i in range(batchlen):
            axarr[i,0].imshow(image_batch[i],cmap='gray')
            axarr[i,1].imshow(predicted_mask_batch[i],cmap='gray')
            for num,box in enumerate(predicted_boxes_batch[i]):
                if np.all(np.equal(box,0)):
                    continue
                else:
                    axarr[i,0].plot(*draw_bbox(box),linewidth=2, alpha=1, color='pink')
                    axarr[i,1].text(box[1]+70,box[0]-5,predicted_scores_batch[i][num],color='pink',fontsize=12)
                    axarr[i,1].text(box[1],box[0]-5,classdict[predicted_label_batch[i][num]],color='pink',fontsize=12)
                    axarr[i,1].plot(*draw_bbox(box),linewidth=2, alpha=1, color='pink')
                    
            if gt_mask_batch is not None:           
                axarr[i,2].imshow(gt_mask_batch[i],cmap='gray')

                    
    else:
        if gt_mask_batch is None:
            f, axarr = plt.subplots(1,2)
        else:
            f, axarr = plt.subplots(1,3)  
            
        axarr[0].imshow(image_batch[0],cmap='gray')
        axarr[1].imshow(predicted_mask_batch[0],cmap='gray')
        for num,box in enumerate(predicted_boxes_batch[0]):
            if np.all(np.equal(box,0)):
                continue
            else:
                axarr[0].plot(*draw_bbox(box),linewidth=2, alpha=1, color='pink')
                axarr[1].text(box[1]+50,box[0]-5,predicted_scores_batch[0][num],color='pink',fontsize=12)
                axarr[1].text(box[1],box[0]-5,classdict[predicted_label_batch[0][num]],color='pink',fontsize=12)
                axarr[1].plot(*draw_bbox(box),linewidth=2, alpha=1, color='pink')
                
        if gt_mask_batch is not None:           
            axarr[2].imshow(gt_masks_batch[0],cmap='gray')


def nms(boxes, scores, proposal_count=20,nms_threshold=0.7,padding=True):

    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(boxes, scores, proposal_count, iou_threshold=0.5)
    proposals = tf.gather(boxes, selected_indices)
    proposal_scores= tf.gather(scores, selected_indices)
    if padding:
    # Pad if needed
        padding = tf.maximum(proposal_count - tf.shape(selected_indices)[0], 0)
        proposals = tf.pad(proposals, [(0, padding), (0, 0)])
    return proposals,proposal_scores,selected_indices


                
def get_proposals(batch_of_pred_scores,batch_of_pred_deltas,anchors,proposal_count=20,mode='pixelwise'):
    
    batchlen=batch_of_pred_scores.shape[0]
    proposals=np.zeros((batchlen,proposal_count,4))
    origanchors=np.zeros((batchlen,proposal_count,4))
    
    for image in range(batchlen):
        pred_scores=batch_of_pred_scores[image]
        pred_deltas=batch_of_pred_deltas[image]
        # Find where predicted positive boxes
        
        positive_idxs = np.where(np.argmax(pred_scores, axis=-1)==1)[0]
        positive_anchors=anchors[positive_idxs]
        selected_boxes=tf.gather(pred_deltas,positive_idxs)
        selected_scores=tf.gather(pred_scores,positive_idxs)
        selected_scores=selected_scores[:,1]
                
        # Get the predicted anchors for the positive anchors
        if mode=='pixelwise':
            predicted_boxes = shift_bbox_pixelwise(positive_anchors, selected_boxes)
        else:
            predicted_boxes = shift_bbox_exponential(positive_anchors, selected_boxes)
            
        sorted_indices=tf.argsort(selected_scores,direction='DESCENDING')
        sorted_boxes=tf.cast(tf.gather(predicted_boxes,sorted_indices),tf.float32)
        sorted_scores=tf.gather(selected_scores,sorted_indices)
        #sorted_anchors=tf.cast(tf.gather(positive_anchors,sorted_indices),tf.float32)

        
        proposals[image],_,_=nms(sorted_boxes,sorted_scores,proposal_count)
        #origanchors[image]=nms(sorted_anchors,sorted_scores,proposal_count)
        
    return proposals


def freeze(model):
    for l in model.layers:
        l.trainable = False

def unfreeze(model):
    for l in model.layers:
        l.trainable = True
        
        
def roi_align(batch_of_featuremaps, proposals, size):
    batchlen=proposals.shape[0]
    proposal_count=proposals.shape[1]
    depth=batch_of_featuremaps.shape[-1]
    allrois=np.zeros((batchlen,proposal_count,size[0],size[1],depth))
    for image in range(batchlen):
        featuremap=batch_of_featuremaps[image:image+1]
        proposal=proposals[image]
        proposal=proposal[:]/512
        allrois[image] = tf.image.crop_and_resize(featuremap, proposal,tf.zeros([tf.shape(proposal)[0]], dtype=tf.int32),size)
    return allrois


def mask_roi_align(batch_of_featuremaps,batch_of_masks, proposals, size):
    batchlen=proposals.shape[0]
    proposal_count=proposals.shape[1]
    depth=batch_of_featuremaps.shape[-1]
    
    mask_size=[size[0]*2,size[1]*2]
    
    allrois=np.zeros((batchlen,proposal_count,size[0],size[1],depth))
    maskrois=np.zeros((batchlen,proposal_count,mask_size[0],mask_size[1],1))
    for image in range(batchlen):
        featuremap=batch_of_featuremaps[image:image+1]
        mask=batch_of_masks[image:image+1]
        proposal=proposals[image]
        proposal=proposal[:]/512
        allrois[image] = tf.image.crop_and_resize(featuremap, proposal,tf.zeros([tf.shape(proposal)[0]], dtype=tf.int32),size)
        maskrois[image] = tf.image.crop_and_resize(mask, proposal,tf.zeros([tf.shape(proposal)[0]], dtype=tf.int32),mask_size)
    return allrois,maskrois


def adjust_mask(foreground_proposals,predicted_masks,preds_per_images,batchlen,origsize=512):
    foreground_proposals=np.rint(foreground_proposals).astype(np.int32)
    fullimage_masks=np.zeros((batchlen,origsize,origsize,1),dtype=np.float32)
    num=0
    for im in range(batchlen):
        for proposal in range(preds_per_images[im]):
            if preds_per_images[im]==0:
                continue
            actualproposal=proposal+num
            x1=foreground_proposals[actualproposal,0]
            x2=foreground_proposals[actualproposal,2]
            y1=foreground_proposals[actualproposal,1]
            y2=foreground_proposals[actualproposal,3]

            #if the predicted anchor is on the edge of the image, we need to cut it to be on the image
            if x1<0:
                x1=0
            if y1<0:
                y1=0
            if x2>511:
                x2=511
            if y2>511:
                y2=511

            size=[x2-x1,y2-y1]
            
            if np.all(np.equal(size,0)):
                continue
            mask=np.copy(predicted_masks[actualproposal])
            mask=tf.image.resize_with_pad(mask, size[0], size[1])
            fullimage_masks[im][x1:x2,y1:y2,:]=mask
            
        num+=preds_per_images[im]
    return fullimage_masks
  
def adjust_boxes(predicted_refined_boxes,preds_per_images,batchlen):
    num=0
    adjusted_boxes=[]
    for im in range(batchlen):
        first=num
        last=first+preds_per_images[im]
        
        rounded=np.rint(predicted_refined_boxes[first:last])
        adjusted_boxes.append(rounded)
        num+=preds_per_images[im]
    return adjusted_boxes
  
def adjust_scores(predicted_softmax_scores,preds_per_images,batchlen):
    num=0
    adjusted_scores=[]
    adjusted_labels=[]
    for im in range(batchlen):
        first=num
        last=first+preds_per_images[im]
        adjusted_scores.append(np.amax(predicted_softmax_scores[first:last],axis=-1))
        adjusted_labels.append(np.argmax(predicted_softmax_scores[first:last],axis=-1))
        num+=preds_per_images[im]
        
    return adjusted_scores,adjusted_labels
    
def adjust_to_batch(foreground_proposals,predicted_classlabels,predicted_softmax_scores,predicted_refined_boxes,preds_per_images,batchlen,predicted_masks=None):
    adjusted_boxes=adjust_boxes(predicted_refined_boxes,preds_per_images,batchlen)
    adjusted_scores,adjusted_labels=adjust_scores(predicted_softmax_scores,preds_per_images,batchlen)
    
    for i in range(len(adjusted_scores)):
        box_proposals,score_proposals,indices=nms(adjusted_boxes[i],adjusted_scores[i],nms_threshold=0.8,padding=False)
        label_proposals=tf.gather(adjusted_labels[i],indices)
        adjusted_boxes[i]=box_proposals.numpy()
        adjusted_scores[i]=np.around(score_proposals.numpy(),decimals=3)
        adjusted_labels[i]=label_proposals.numpy()
        
    if predicted_masks is None:
        return adjusted_boxes,adjusted_scores,adjusted_labels
    else:
        adjusted_masks=adjust_mask(foreground_proposals,predicted_masks,preds_per_images,batchlen)
        return adjusted_masks,adjusted_boxes,adjusted_scores,adjusted_labels

