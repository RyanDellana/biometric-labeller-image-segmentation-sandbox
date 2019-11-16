# I'd eventually like to move everything non-gui into this file so that BLISS.pyw contains
# only gui/view code.

import cv2
from cv2 import cv
from PIL import Image

import numpy as np
import math, time

try:
    import eyeLike
except ImportError:
    raise ImportError('Could not import the eyeLike module. Did you run make?')


class Eye(object):

    def __init__(self, cascade_box=None, center=None):
        self.confidence = 0.0           # confidence level.
        self.cascade_box = cascade_box  # cascade classifier bounding box.
        self.contour = None             # contour obtained from grabcut segmentation.
        self.center = center            # center of min enclosing circle.
        self.radius = None              # radius of min enclosing circle.
        self.img = None                 # image taken from circle bounding box.
        self.img_unrolled = None        # unrolled rectangular image.

    def unroll(self):
        if self.img_unrolled is None and self.img is not None:
            pass # TODO unroll self.img using some algorithm TODO ??


class Face(object):

    def __init__(self, cascade_box=None, left_eye=None, right_eye=None, confidence=0.0, ts=None):
        self.confidence = confidence   # detection confidence.
        self.left_eye = left_eye       # left eye object.
        self.right_eye = right_eye     # right eye object.
        self.cascade_box = cascade_box # box from cascade detector.
        self.img_cascade_box = None    # face image from composite box.
        self.img_cropped = None        # cropped and rotated image suitable for matching.
        self.label = None              # intra-class label (i.e. who's face this is)
        self.label_confidence = 0.0    # confidence of label (duh)
        # --------- Used in Temporal Mode ----------------
        if ts is None:
            self.ts = time.time()          # timestamp of current data.
        else:
            self.ts = ts
        self.ts_prev = None            # timestamp of previous data.
        self.cascade_box_history = []  # previous cascade boxes (ts, cascade_box)
        self.label_history = []        # previous labels (ts, label, label_confidence)        
        self.eye_history = []          # (ts, left_eye, right_eye)
        self.identified = False
        self.detected = False
        if cascade_box is not None:
            self.cascade_box_history.append((self.ts, cascade_box))
        if left_eye is not None and right_eye is not None:
            self.eye_history.append((self.ts, left_eye, right_eye))
        # -------- Not currently in use ------------------
        self.contour = None            # face contour from grabCut.
        self.contour_box = None        # contour bounding box.
        self.composite_box = None      # contour_box constrained by cascade_box.
        self.hist_rotated_rect = None  # rotated rectangle from cam-shift of histogram back-projection.
        self.img_lbp = None            # local binary pattern image.

    # return euclidean distance.
    def __sub__(self, other):
        if None in [self.position(), self.size(), other.position(), other.size()]:
            return None
        else:
            (x, y) = self.position()
            (x2, y2) = other.position()
            return math.sqrt((x-x2)**2 + (y-y2)**2 + (self.size()-other.size())**2)

    # ------------- Temporal Functions -----------------------------

    # update face position with a new cascade_box or left_eye/right_eye.
    # you can call update with no parameters if you nothing was detected.
    def update(self, cascade_box=None, left_eye=None, right_eye=None, confidence=0.0, label=None, label_conf=0.0, ts=None):
        #print str(self), 'update(', cascade_box, left_eye, right_eye, confidence, label, label_conf, ts, ')'
        if ts is None:
            self.ts_prev = self.ts
            self.ts = time.time()
        elif ts != self.ts: # new time.
            self.ts_prev = self.ts
            self.ts = ts
        # always updated, even when there's no detection.
        if cascade_box is not None:
            self.cascade_box = cascade_box
        self.cascade_box_history.append((self.ts, cascade_box))
        if left_eye is not None and right_eye is not None:
            self.left_eye, self.right_eye = left_eye, right_eye
            self.eye_history.append((self.ts, left_eye, right_eye))
        if label is not None:
            self.label_history.append((self.ts, label, label_conf))
        # memory management.
        if len(self.cascade_box_history) > 100:
            del self.cascade_box_history[0]
        if len(self.eye_history) > 100:
            del self.eye_history[0]
        if len(self.label_history) > 100:
            del self.label_history[0]
        #print 'self.label_history:', self.label_history
        #print 'len(self.cascade_box_history):', len(self.cascade_box_history)
        # update detected.
        if not self.detected: # initial detection criteria.
            # 3 consequtive detections to detect.
            if len(self.cascade_box_history) >= 2:
                recent3 = self.cascade_box_history[-3:]
                if not (None in [i[1] for i in recent3]):
                    self.detected = True
        """
        else: # still have a lock?
            # 1 detection in most recent 5 frames.
            if len(self.cascade_box_history) >= 5:
                recent5 = self.cascade_box_history[-5:]
                cnt = 0
                for f in recent5:
                    if f[1] == None:
                        cnt += 1
                if cnt == 5:
                    self.detected = False
                    # self.identified = False
        """
        # update identity
        if not self.identified:
            if len(self.label_history) >= 5:
                recent = self.label_history[-5:]
                recent = [i[1] for i in recent]
                # print 'recent:', recent
                cnt = Counter(recent)
                (lbl, votes) = cnt.most_common(1)[0]
                if votes == 5:
                    self.identified = True
                    self.label = lbl
                    self.label_confidence = votes/5.0

    # time since last tracking update.
    def age(self):
        for i in range(len(self.cascade_box_history)-1, -1, -1):
            (ts, box) = self.cascade_box_history[i]
            if box is not None:
                return time.time() - ts
        return 60.0 # default to one minute.

    # who the face is. None if not enough information.
    def identity(self):
        if self.identified:
            return self.label
        else:
            return None

    # where the face is.
    def position(self):
        if self.cascade_box is not None:
            (x, y, w, h) = self.cascade_box
            return (x+w/2, y+h/2)
        else:
            return None

    # how large the face is.
    def size(self):
        if self.cascade_box is not None:
            (x, y, w, h) = self.cascade_box
            return w
        else:
            return None

    # where the face is going.
    def vector(self):
        pass

    # --------------------------------------------------------------

    # crop, rotate, and scale using eyes as reference points.
    def normalize(self):
        if None in [self.cascade_box, self.left_eye, self.right_eye]:
            return None
        if self.img_cropped is not None:
            return self.img_cropped
        (x, y, w, h) = self.cascade_box
        (lx, ly), (rx, ry) = self.left_eye.center, self.right_eye.center
        leye_center_, reye_center_ = (lx-x, ly-y), (rx-x, ry-y) # convert from img coordinates to face_box coordinates
        self.img_cropped = self.cropFace(self.img_cascade, leye_center_, reye_center_, dest_sz=(256,256))
        return self.img_cropped

    # Guess iris region from face dimensions
    # returns an image with the irises circled.
    def segment_irises_from_cropped(self):
        if self.img_cropped is None:
            return None
        face_gray = cv2.cvtColor(self.img_cropped, cv2.COLOR_BGR2GRAY)
        (h__, w__) = face_gray.shape
        eye_w = int(w__/5.0)
        lx__, ly__ = int(w__*0.25)-eye_w/2, int(h__*0.25)-eye_w/2
        lx2__, ly2__ = lx__+eye_w, ly__+eye_w
        rx__, ry__ = int(w__*0.75)-eye_w/2, int(h__*0.25)-eye_w/2
        rx2__, ry2__ = rx__+eye_w, ry__+eye_w
        leye_roi, reye_roi = (lx__, ly__, eye_w, eye_w), (rx__, ry__, eye_w, eye_w)
        (lx_, ly_), (rx_, ry_) = self.findEyes(face_gray, (0,0,face_gray.shape[0],face_gray.shape[1]), leye_roi, reye_roi)
        img = self.img_cropped.copy()
        cv2.circle(img, (int(lx_), int(ly_)), int(round(eye_w/4.0)), (0,255,0), 1)
        cv2.circle(img, (int(rx_), int(ry_)), int(round(eye_w/4.0)), (0,255,0), 1)
        return img

    def lbp_img(self):
        if self.img_cropped is None:
            return None
        if self.img_lbp is not None:
            return self.img_lbp
        radius = 3 # Number of points to be considered as neighbourers
        no_points = 8 * radius # Uniform LBP is used 
        gray = cv2.cvtColor(self.img_cropped, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, no_points, radius, method='uniform')
        self.img_lbp = lbp
        return self.img_lbp

    def findEyes(self, img_gray, face, left_eye, right_eye):
        (x, y, w, h) = face
        (x1, y1, w1, h1) = left_eye
        (x2, y2, w2, h2) = right_eye
        faceROI = img_gray[y:y+h,x:x+w].copy()
        faceROI = cv2.equalizeHist(faceROI)
        leftEyeRegion = np.array([[float(x1-x)],[float(y1-y)],[float(w1)],[float(h1)]])
        rightEyeRegion = np.array([[float(x2-x)],[float(y2-y)],[float(w2)],[float(h2)]])
        leftPupil, rightPupil = [[x1+w1/2],[y1+h1/2]], [[x2+w2/2],[y2+h2/2]] # default in-case functions throw errors.
        try:
            leftPupil = eyeLike.findEyeCenter(faceROI, leftEyeRegion)
            rightPupil = eyeLike.findEyeCenter(faceROI, rightEyeRegion)
        except:
            pass
        leftPupil_x, leftPupil_y = leftPupil[0][0], leftPupil[1][0]
        rightPupil_x, rightPupil_y = rightPupil[0][0], rightPupil[1][0]
        rightPupil_x += rightEyeRegion[0][0]
        rightPupil_y += rightEyeRegion[1][0]
        leftPupil_x += leftEyeRegion[0][0]
        leftPupil_y += leftEyeRegion[1][0]
        return (leftPupil_x, leftPupil_y), (rightPupil_x, rightPupil_y)

    # http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html
    def distance(self, p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.sqrt(dx*dx+dy*dy)

    # http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html
    def scaleRotateTranslate(self, image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
        if (scale is None) and (center is None):
            return image.rotate(angle=angle, resample=resample)
        nx,ny = x,y = center
        sx=sy=1.0
        if new_center:
            (nx,ny) = new_center
        if scale:
            (sx,sy) = (scale, scale)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        a = cosine/sx
        b = sine/sx
        c = x-nx*a-ny*b
        d = -sine/sy
        e = cosine/sy
        f = y-nx*d-ny*e
        return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)

    # image is the face roi (cascade_box)
    # http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html
    def cropFace(self, image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.23,0.23), dest_sz=(256,256)):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        offset_h = math.floor(float(offset_pct[0])*dest_sz[0]) # calculate offsets in original image
        offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
        eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1]) # get the direction
        rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0])) # calc rotation angle in radians
        dist = self.distance(eye_left, eye_right) # distance between them
        reference = dest_sz[0] - 2.0*offset_h # calculate the reference eye-width
        scale = float(dist)/float(reference) # scale factor
        image = self.scaleRotateTranslate(image, center=eye_left, angle=rotation) # rotate original around the left eye
        crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v) # crop the rotated image
        crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
        image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
        image = image.resize(dest_sz, Image.ANTIALIAS) # resize it
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy() # Convert RGB to BGR 
        return open_cv_image

    def draw_cropped(self, img):
        self.overlay(img, self.img_cropped)

    def draw_lbp(self, img):
        lbp = cv2.equalizeHist(self.img_lbp.astype(np.uint8))
        lbp = cv2.cvtColor(lbp, cv2.COLOR_GRAY2BGR)
        self.overlay(img, lbp)

    def overlay(self, img, overlay_img):
        (x, y, w, h) = self.cascade_box
        scaled = cv2.resize(overlay_img, (h, h), interpolation=cv2.INTER_CUBIC)
        (h_, w_, _) = scaled.shape
        x2 = x + (w-w_)/2
        img[y:y+h, x2:x2+w_] = scaled


# Face Detector Identifier/Recognizer and Tracker
# Face DIRT lol!
class FaceDRT(object):

    def __init__(self, temporal_mode=False):
        self.img = None
        self.img_gray = None
        self.img_small = None
        self.img_small_gray = None
        self.small_width = 160
        self.faces = []
        self.face = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt.xml')
        self.eye = cv2.CascadeClassifier('./haarcascades/haarcascade_eye_tree_eyeglasses.xml')
        self.interp = cv2.INTER_CUBIC # cv2.INTER_NEAREST # cv2.INTER_LINEAR cv2.INTER_AREA cv2.INTER_CUBIC cv2.INTER_LANCZOS4
        self.id_label = {}
        self.label_id = {}
        self.training_faces = []
        self.mugshot = {}
        # self.recognizerLBPH = cv2.createLBPHFaceRecognizer()
        # self.recognizerEigen = cv2.createEigenFaceRecognizer()
        # self.recognizerFisher = cv2.createFisherFaceRecognizer()
        self.temporal_mode = temporal_mode
        self.max_facial_velocity = 1.0 # max percentage of width change in position per second
        # self.load_training()

    # min_size = minimum size of face as percentage of image width.
    # small_width = pixel width of shrunk image used to speed up detection.
    def detect(self, img, draw=False, min_size=0.10, small_width=160):
        self.small_width = small_width
        faces_ = [] # faces detected in this round.
        if self.temporal_mode == False:
            self.faces = []
        self.img = img.copy()
        (img_h, img_w, _) = self.img.shape
        if small_width < img_w:
            self.img_small = cv2.resize(self.img, (small_width, int(img_h*(float(small_width)/img_w))), interpolation=self.interp)
        else:
            self.img_small = img.copy()
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img_small_gray = cv2.cvtColor(self.img_small, cv2.COLOR_BGR2GRAY)
        (img_s_h, img_s_w) = self.img_small_gray.shape
        scale = int(round(img_w/float(img_s_w)))
        min_size_px = int(img_s_w * min_size)
        face_candidates = self.face.detectMultiScale(self.img_small_gray, 1.1, 2, minSize=(min_size_px,min_size_px))
        if face_candidates is not None and len(face_candidates) > 0:
            for face_box_small in face_candidates:
                (x, y, w, h) = face_box_small
                x, y, w, h = x*scale, y*scale, w*scale, h*scale
                face_box = (x, y, w, h)
                x_, y_, w_, h_ = x+int(0.08*w), y+int(0.20*h), int(w*0.84), int(h*0.38) #int(h*0.55)
                eye_min, eye_max = int(h*0.12), int(h*0.25) #int(h*0.15), int(h*0.25)
                eye_search_roi = self.img_gray[y_:y_+h_,x_:x_+w_].copy()
                eye_search_roi = cv2.equalizeHist(eye_search_roi)
                eye_candidates = self.eye.detectMultiScale(eye_search_roi, 1.025, 1, minSize=(eye_min,eye_min), maxSize=(eye_max,eye_max))
                eye_candidates = [list(i) for i in eye_candidates]
                eye_candidates = [(x_+xi, y_+yi, wi, hi) for (xi, yi, wi, hi) in eye_candidates]
                left_eyes, right_eyes = [], [] # detection on left and right side.
                left_eye, right_eye = None, None
                face_midline = (x+w/2)
                for eye in eye_candidates:
                    if eye[0]+eye[2]/2 >= face_midline: # right eye
                        right_eyes.append(eye)
                    else: # left eye
                        left_eyes.append(eye)
                left_eyes = self.remove_extra_eyes(left_eyes, face_box)
                right_eyes = self.remove_extra_eyes(right_eyes, face_box)
                leye_conf, reye_conf = len(left_eyes)*1.0, len(right_eyes)*1.0
                confidence = 0.333 + (len(left_eyes)+len(right_eyes))*0.333
                if len(left_eyes)+len(right_eyes) == 2:
                    # set the size of each to their average.
                    (x1,y1,w1,h1), (x2,y2,w2,h2) = left_eyes[0], right_eyes[0]
                    (x1_, y1_) = x1+w1/2, y1+h1/2 # center
                    (x2_, y2_) = x2+w2/2, y2+h2/2 # center
                    w_avg, h_avg = (w1+w2)/2, (h1+h2)/2
                    left_eyes, right_eyes = [(x1_-w_avg/2,y1_-h_avg/2,w_avg,h_avg)], [(x2_-w_avg/2,y2_-h_avg/2,w_avg,h_avg)]
                elif len(left_eyes)+len(right_eyes) == 1:
                    # auto-generate other eye as mirror of detected eye. TODO don't do this if eye is too close to center. TODO
                    (x1,y1,w1,h1) = (left_eyes+right_eyes)[0]
                    (x1_, y1_) = x1+w1/2, y1+h1/2 # center
                    dist_from_midline = abs(face_midline - x1_)
                    if x1_ < face_midline: # generate right eye.
                        right_eyes.append(((face_midline+dist_from_midline-w1/2)-int(w1*0.1), y1-int(h1*0.1), int(w1*1.2), int(h1*1.2)))
                    else: # generate left eye
                        left_eyes.append(((face_midline-dist_from_midline-w1/2)-int(w1*0.1), y1-int(h1*0.1), int(w1*1.2), int(h1*1.2)))
                else: # generate default eye regions.  
                    eye_width = int(w*0.30) # eye region width is 30% the width of the face.
                    yi = int(0.38*h - eye_width/2.0)+y
                    left_eyes = [(int((x+0.30*w)-eye_width/2.0), yi, eye_width, eye_width)]
                    right_eyes = [(int((x+0.69*w)-eye_width/2.0), yi, eye_width, eye_width)]
                ts = time.time()
                face = Face(cascade_box=face_box, confidence=confidence, ts=ts)
                face.img_cascade = self.img[y:y+h, x:x+w].copy()
                left_eye_box, right_eye_box = left_eyes[0], right_eyes[0]
                (lx, ly), (rx, ry) = face.findEyes(self.img_gray, (x_,y_,w_,h_), left_eye_box, right_eye_box)
                lx, ly, rx, ry = x_+int(lx), y_+int(ly), x_+int(rx), y_+int(ry)
                leye, reye = Eye(left_eye_box, (lx,ly)), Eye(right_eye_box, (rx,ry))
                leye.confidence, reye.confidence = leye_conf, reye_conf
                leye.center, reye.center = (lx, ly), (rx, ry)
                leye.radius, reye.radius = None, None # guess based on distance eyes are apart.
                if self.temporal_mode:
                    face.update(left_eye=leye, right_eye=reye, ts=ts)
                else:
                    face.left_eye, face.right_eye = leye, reye
                # TODO Disabled face recognition post-validation. Need a substitute for this. TODO
                #if confidence < 0.5: # no eyes detected. Is this a face?
                # lbph_conf, fisher_conf, eigen_conf = self._identify(face, ts=ts)
                # if lbph_conf > 100.0 or eigen_conf < 7000.0:
                #     continue # reject if low confidence.
                #    # if it doesn't look at all like anyone, then it probably isn't a face.
                faces_.append(face)
        if len(self.faces) == 0:
            self.faces = faces_
        else: # only active when temporal_mode == True
            new_faces = self.pair_faces(faces_)
            self.faces.extend(new_faces)
        if draw:
            self.draw_all(img)
        return self.faces

    # Updates existing faces and returns new faces.
    def pair_faces(self, faces):
        (img_h, img_w, _) = self.img.shape
        face_pairs = [[None, face] for face in faces]
        for face in self.faces:
            closest = None
            closest_dist = 100000.0
            closest_idx = 0
            for idx, pairing in enumerate(face_pairs):
                (a, b) = pairing
                if a is None:
                    if (face - b) < closest_dist:
                        closest_idx = idx
                        closest_dist = face - b
                        closest = b
            if closest is not None:
                face_pairs[closest_idx][0] = face
        # Now we have the pairings.
        new_faces = []
        for pair in face_pairs:
            (old_face, new_face) = pair
            if old_face is None:
                new_faces.append(new_face)
            else: # Is it close enough to be considered the same face?
                interval = new_face.ts - old_face.ts
                radius = self.max_facial_velocity * interval
                normed_dist = (old_face - new_face)/float(img_w)
                if normed_dist <= radius: # same face
                    #print 'updating old_face with', str(new_face), new_face.label
                    new_lbl, new_lbl_conf = None, 0.0
                    if len(new_face.label_history) > 0:
                        (_, new_lbl, new_lbl_conf) = new_face.label_history[0]
                    old_face.update(cascade_box=new_face.cascade_box, left_eye=new_face.left_eye, right_eye=new_face.right_eye, confidence=new_face.confidence, label=new_lbl, label_conf=new_lbl_conf, ts=new_face.ts)
                else:
                    new_faces.append(new_face)
        return new_faces

    # identify all faces detected in self.img
    # If img is passed, draw face in img.
    def identify(self, img=None):
        for face in self.faces:
            self._identify(face, img)

    def _identify(self, face, img=None, ts=None):
        cropped = cv2.cvtColor(face.normalize(), cv2.COLOR_BGR2GRAY)
        if cropped is not None:
            lbph_label, lbph_conf = self.recognizerLBPH.predict(cropped)
            fisher_label, fisher_conf = self.recognizerFisher.predict(cropped)
            eigen_label, eigen_conf = self.recognizerEigen.predict(cropped)
            #print 'lbph:  ', lbph_label, lbph_conf
            #print 'fisher:', fisher_label, fisher_conf
            #print 'eigen: ', eigen_label, eigen_conf
            lbl, votes = '', 0
            cnt = Counter([lbph_label, fisher_label, eigen_label])
            if len(cnt.most_common()) != 3:   # Returns all unique items and their counts
                (lbl, votes) = cnt.most_common(1)[0]      # Returns the highest occurring item
            else:
                lbl, votes = eigen_label, 1
            face.update(label=self.id_label[lbl], label_conf=votes/3.0, ts=ts)
            if img is not None:
                face.overlay(img, self.mugshot[face.label])
            return (lbph_conf, fisher_conf, eigen_conf)
        else:
            return None

    def load_training(self):
        if len(self.training_faces) > 0:
            return True
        path = engrams_path + 'face_recognition/'
        try:
            self.recognizerLBPH.load(path+'lbph')
            self.recognizerFisher.load(path+'fisher')
            self.recognizerEigen.load(path+'eigen')
            with open(path+'training_data.pickle','r') as FILE:
                data = cPickle.load(FILE)
                (self.label_id, self.id_label, self.training_faces, self.training_labels) = data
            for idx, face in enumerate(self.training_faces):
                #print idx, self.training_labels[idx]
                self.mugshot[self.training_labels[idx]] = face
            return True
        except:
            return False

    def save_training(self):
        path = engrams_path + 'face_recognition/'
        self.recognizerLBPH.save(path+'lbph')
        self.recognizerFisher.save(path+'fisher')
        self.recognizerEigen.save(path+'eigen')
        with open(path+'training_data.pickle','w') as FILE:
            cPickle.dump((self.label_id, self.id_label, self.training_faces, self.training_labels), FILE, protocol=2)
        
    # Function to train faceTracker on a face database.
    # faces = list of image arrays all the same size.
    # labels = list of strings which serve as labels for the faces.
    def train(self, faces, labels):
        faces_gray = []
        if len(faces) > 0:
            if len(faces[0].shape) == 3: # if color images
                for face in faces:
                    faces_gray.append(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))
            else:
                faces_gray = faces
        next_id = len(self.id_label.keys())
        for label in labels:
            if label not in self.label_id:
                self.id_label[next_id] = str(label)
                self.label_id[str(label)] = next_id
                next_id += 1
        labels_int = np.array([self.label_id[label] for label in labels])
        self.training_faces = faces
        self.training_labels = labels
        self.recognizerLBPH.train(faces_gray, labels_int)
        self.recognizerFisher.train(faces_gray, labels_int)
        self.recognizerEigen.train(faces_gray, labels_int)

    # TRACKING TRICKS;
    # 1. Dynamic cascade.
    # given the time between frames, you can get constraints on the change in scale and position of the face.
    # this allows a region of interest and min/max size parameters to be tweaked for performance from
    # frame to frame. You also don't need to track the eyes since the previous detection provides confidence.
    # 2. Histogram back-projection + Cam-shift may be useful.
    # 3. Texture-Features + Optical Flow: (see demo)
    # gives the benefit of head pose.
    # 4. Probably best to combine all three of the above.
    # Motion-based background subtraction may also prove useful when the camera is static.
    # TODO To make all this run smoothly we need multiprocessing.
    # ----------------------
    # Track one face at a time for now.
    # 1. Aquire = call detect repeatedly until you detect a face in an area three frames in a row.
    # 2. Identify = run identify for 10 frames and choose mode identity.
    # 3. Track = just run face cascade and assume same face.
    # ^^^ Do not display tracking info until aquire and identify are finished. Takes a few seconds.
    # TODO TODO faces should be removed if they have not been updated in a while.
    def drt(self, img, timestamp=None):
        self.temporal_mode = True
        ts = time.time()
        if timestamp is not None:
            ts = timestamp
        detected = False
        identified = False
        for face in self.faces:
            if face.detected and face.identified:
                detected, identified = True, True
            elif face.detected:
                detected = True
        if detected == False or identified == False:
            self.faces = self.detect(img, min_size=0.15)
        else: # at least one face has been detected and identified
            faces = self.get_faces_cascade(img, ts)
            new_faces = self.pair_faces(faces)
            for face in self.faces:
                if face.ts != ts:
                    face.update() # update faces not detected.
            #for face in self.faces:
            #    if face.detected:
            #        self.track(img, face, ts)
        # garbage collect lost faces.
        faces_ = []
        for face in self.faces:
            age_limit = 2.0
            if face.detected and face.identified:
                age_limit = 0.5
            if face.age() < age_limit:
                faces_.append(face)
        self.faces = faces_
        return self.faces

    # track face
    def track(self, img, face, timestamp):
        pass # TODO run a targeted cascade.

    def get_faces_cascade(self, img, ts=None):
        self.img = img.copy()
        (img_h, img_w, _) = self.img.shape
        self.img_small = cv2.resize(self.img, (self.small_width, int(img_h*(float(self.small_width)/img_w))), interpolation=self.interp)
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img_small_gray = cv2.cvtColor(self.img_small, cv2.COLOR_BGR2GRAY)
        (img_s_h, img_s_w) = self.img_small_gray.shape
        scale = int(round(img_w/float(img_s_w)))
        min_size_px = int(img_s_w * 0.15)
        face_candidates = self.face.detectMultiScale(self.img_small_gray, 1.1, 2, minSize=(min_size_px,min_size_px))
        face_boxes = [(x*scale, y*scale, w*scale, h*scale) for (x, y, w, h) in face_candidates]
        faces = [Face(cascade_box=box, ts=ts) for box in face_boxes]
        return faces

    def remove_extra_eyes(self, eyes, face_box):
        eyes_ = eyes
        if len(eyes) > 1:
            best_score, eye = self.eye_dist_from_expected(eyes[0], face_box), eyes[0]
            for eye_ in eyes:
                score = self.eye_dist_from_expected(eye_, face_box)
                if score < best_score:
                    best_score = score
                    eye = eye_
            eyes_ = [eye]
        return eyes_

    def draw_all(self, img):
        for face in self.faces:
            self.draw(img, face)

    def draw(self, img, face, mask=None):
        (img_h, img_w, _) = img.shape
        (x,y,w,h) = face.cascade_box
        if face.composite_box is not None:
            (x,y,w,h) = face.composite_box #face.cascade_box
        (lx,ly) = face.left_eye.center
        (rx,ry) = face.right_eye.center
        color = (0,0,255) # red
        if face.detected and face.identified:
            color = (0,255,0) # green
        elif face.detected:
            color = (0,255,255) # yellow
        if face.contour is not None:
            c = Contour(face.contour)
            cv2.drawContours(img, [face.contour], 0, (0,0,255), 3)
            mask_img = mask.copy()
            mask_img = np.where((mask_img==2),100,mask_img).astype('uint8')   # probable background
            mask_img = np.where((mask_img==1),255,mask_img).astype('uint8')  # definite foreground
            mask_img = np.where((mask_img==3),170,mask_img).astype('uint8')  # probable foreground
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
            mask_img = cv2.resize(mask_img, (img_w, img_h), interpolation=self.interp)
            img_ = cv2.addWeighted(img, 0.8, mask_img, 0.2, 0) # TODO this doesn't change img
            img[:,:,:] = img_[:,:,:]
        if face.hist_rotated_rect is not None:
            pts = cv2.cv.BoxPoints(face.hist_rotated_rect)
            pts = np.int0(pts)
            _ = cv2.polylines(img, [pts], True, 255, 2)
            cv2.ellipse(img, box=face.hist_rotated_rect, color=(0,0,255), thickness=3)
        (xi, yi, wi, hi) = face.left_eye.cascade_box
        eyec = (0,255,0) if face.left_eye.confidence == 1.0 else (0,0,255)
        if not self.temporal_mode or not face.detected:
            cv2.rectangle(img,(xi,yi),(xi+wi,yi+hi),eyec,2)
        (xi, yi, wi, hi) = face.right_eye.cascade_box
        eyec = (0,255,0) if face.right_eye.confidence == 1.0 else (0,0,255)
        if not self.temporal_mode or not face.detected:
            cv2.rectangle(img,(xi,yi),(xi+wi,yi+hi),eyec,2)
            # TODO The face can draw it's own irises.
            cv2.circle(img, (lx, ly), int((rx-lx)*0.09), (255,0,0), 2)
            cv2.circle(img, (rx, ry), int((rx-lx)*0.09), (255,0,0), 2)
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        if self.temporal_mode:
            txt = 'detecting...'
            if face.identified:
                txt = 'Identity: ' + face.label
            elif face.detected:
                txt = 'identifying...'
            cv2.putText(img, txt, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
    def eye_dist_from_expected(self, eye, face):
        (x,y,w,h) = face
        (xi,yi,wi,hi) = eye
        xi, yi = xi+wi/2, yi+hi/2 # get center of eye region.
        face_midline = (x+w/2)
        expected_x, expected_y = 0.0, y + 0.38*h
        if xi > face_midline: # get distance from right eye expected.
            expected_x = x + 0.66*w
        else: # get distance from left eye expected.
            expected_x = x + 0.33*w
        return math.sqrt((expected_x-xi)**2.0 + (expected_y-yi)**2.0)
