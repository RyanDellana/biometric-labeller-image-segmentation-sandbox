#!/usr/bin/env python

"""
Graphical App using QT and OpenCV.

[ Load images ] file/folder recursive/non-recursive.
Loaded into list-view.

Perform Eye/face detection loaded dataset.
Images are displayed as they are processed.

[ Dump results to json ] file named [text-box]

Relative path is used as unique id of each image.

list of detected faces/eyes is paired with each image.
store full path of original folder (may change if folder is moved)
defaults to looking in same directory as json file.

[ Load dataset ] load json file into memory.

view each labeled image by selecting in list.

edit options underneath, allows you to label data.
click drag box to indicate face roi.
double-click to indicate eye center position.

[ Save changes ] replace original.

Compare face finder results with labeled set. (difference between sets). <<<

----------------------

Assumptions:
> Files are not renamed, moved, or modified in any way after labeling.
> All image files have unique names. Directory names are not used to differentiate them.

Load Dataset: If directory is selected, recursivly find all images and index by unique name.
  If json file is selected, load metadata and files specified in said json file.
Close Dataset: Closes dataset. Does not auto-save.
Save Dataset: Saves dataset meta-data/labels to a json file with supplied name in top level dataset directory.

Segmenting and labeling images:

Segments will be rectangular regions, or regions of interest ROIs.
Each ROI will have labels attached to it, like "face".
Something of class "face" will have attributes including: subject_id, subject_name, age, gender, head_pose

new line

"""

#! /usr/bin/python
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from future_builtins import *

import cv2
from cv2 import cv
from PIL import Image

import os, sys, time
import platform
import numpy as np
import math
from PyQt4.QtCore import (PYQT_VERSION_STR, QFile, QFileInfo, QSettings,
        QString, QT_VERSION_STR, QTimer, QVariant, QObject, Qt, SIGNAL, SLOT, QLineF, QRectF, QSizeF, QPointF,
        pyqtSignal, pyqtSlot)
from PyQt4.QtGui import (QWidget, QAction, QActionGroup, QApplication, QShortcut, QLayout,
        QFrame, QIcon, QImage, QImageReader, QLabel, QCheckBox, QStatusBar,
        QImageWriter, QInputDialog, QKeySequence, QLabel, QListWidget, QSplitter, QSpinBox, QComboBox, 
        QMainWindow, QMessageBox, QPainter, QPixmap, QPrintDialog, QClipboard,
        QPrinter, QSpinBox, QPushButton, QGraphicsScene, QGraphicsView, QGraphicsItem, 
        QGraphicsLineItem, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsSimpleTextItem,
        QVBoxLayout, QHBoxLayout, QLineEdit, QDialog, QFileDialog, QFont, QListWidgetItem, QListWidget)

__version__ = "1.0.0"


try:
    import ISUDeepLearning.DataMungingUtil as dmu
except ImportError:
    raise ImportError('Could not import ISUDeepLearning.DataMungingUtil. Some functionality will be unavailable.')

from BLISS.bliss_utils import Eye, Face, FaceDRT


""" handles drag-drop event. Generates ROI_selected signal.
    Returns selected region as normalized image coordinates.
"""

class CVImageGraphicsItem(QGraphicsPixmapItem):

    #regionSelectedSignal = pyqtSignal()

    def __init__(self, parent=None, cvImg=None):
        super(QGraphicsPixmapItem, self).__init__(parent)
        # QObject.__init__(self)
        self.cvImg = None
        self.QImg = None
        self.pxmap = None
        self.updateImg(cvImg)
        self.pressedX = None
        self.pressedY = None

    #def RegisterSignal(self, obj):
    #    self.regionSelectedSignal.connect(obj)

    def updateImg(self, img):
        if img != None:
            self.cvImg = img
            self.QImg = self.cv2QImage(self.cvImg)
            self.pxmap = QPixmap.fromImage(self.QImg)
            self.setPixmap(self.pxmap)
            self.setPos(QPointF(0, 0)) # x, y

    def cv2QImage(self, cv_img):
        if len(cv_img.shape) == 2: # black and white image
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
        height, width, bytesPerComponent = cv_img.shape
        bytesPerLine = bytesPerComponent * width;
        imgcpy = cv_img.copy()
        cv2.cvtColor(imgcpy, cv.CV_BGR2RGB, imgcpy)
        return QImage(imgcpy.data, width, height, bytesPerLine, QImage.Format_RGB888)

    def mousePressEvent(self, event):
        # print("clicked x = " + str(float(event.pos().x())) + " y = " + str(float(event.pos().y())))
        self.pressedX = float(event.pos().x())
        self.pressedY = float(event.pos().y())

    def mouseReleaseEvent(self, event):
        # print("released" + str(event.pos()))
        x = float(event.pos().x())
        y = float(event.pos().y())
        roi_px = [self.pressedX, self.pressedY, 0.0, 0.0]
        if self.pressedX is not None:
            if x > self.pressedX and y > self.pressedY: # only allows selecting from upper left corner. TODO
                #roi_px[2] = x - self.pressedX
                #roi_px[3] = y - self.pressedY
                # convert roi_px to normalized coordinates.
                #normHeight =  self.pixmap().height() / self.pixmap().width() # assume width >= height. TODO
                #roi_px[0] = roi_px[0] / float(self.pixmap().width())
                #roi_px[1] = roi_px[1] / float(self.pixmap().height()) * normHeight
                #roi_px[2] = roi_px[2] / float(self.pixmap().width())
                #roi_px[3] = roi_px[3] / float(self.pixmap().height()) * normHeight
                # print(str(roi_px))
                # generate ROI_selected signal.
                # self.emit(SIGNAL("roiSelected"), self.roi_px)
                # -----------------
                roi_px = (int(self.pressedX), int(self.pressedY), int(x-self.pressedX), int(y-self.pressedY))
                self.roiSelected(roi_px)

    def hoverMoveEvent(self, event):
        print('mouse_move = '+str(event))

    def mouseDoubleClickEvent(self, event):
        print('double_click ='+str(event))
        x = int(event.pos().x())
        y = int(event.pos().y())
        self.findDestination().doubleClickPic((x,y))

    # TODO This is very bad practice. Find the correct way.
    def findDestination(self):
        return self.scene().parent().parent().parent().parent()

    def roiSelected(self, roi_px):
        self.findDestination().roiSelected(roi_px)
        #print('emited regionSelected')
        #self.regionSelectedSignal.emit() # .emit(roi_px)

class CentralWidget(QWidget):

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)
        self.__controls()
        self.__layout()

    def __controls(self):
        self.formWidget = FormWidget()
        self.labelList = LabelList()

    def __layout(self):
        self.vbox = QVBoxLayout()
        self.split = QSplitter()
        self.split.setOrientation(Qt.Vertical)
        self.split.addWidget(self.formWidget)
        self.split.addWidget(self.labelList)
        self.vbox.addWidget(self.split)
        self.setLayout(self.vbox)

class LabelList(QWidget):

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)
        self.__controls()
        self.__layout()
        self.lbls = {}

    def __controls(self):
        pass

    def __layout(self):
        self.vbox = QVBoxLayout()
        self.hbox_lbl_props = QHBoxLayout()
        self.vbox.addLayout(self.hbox_lbl_props)
        self.setLayout(self.vbox)

    def add_lbl(self, lbl, props):
        vbox, lnEdits = self.make_class_vbox(lbl, props)
        self.hbox_lbl_props.addLayout(vbox)
        self.lbls[lbl] = lnEdits

    # returns vbox layout populated with class attribute/value pairs (each being an hbox)
    # label is the class name and props is the dictionary of key-value pairs (properties).
    def make_class_vbox(self, label='label/class', props={'a':0,'b':1,'c':2}):
        print('make_class_vbox '+str(label)+' '+str(props))
        vbox = QVBoxLayout()
        vbox.setAlignment(Qt.Alignment(Qt.AlignTop | Qt.AlignLeft))
        vbox.addWidget(QLabel(label))
        lnEdits = []
        for k in props:
            lnEdit = QLineEdit(k + '=' + str(props[k]))
            vbox.addWidget(lnEdit)
            lnEdits.append(lnEdit)
        btnSave = QPushButton('Save')
        btnSave.setToolTip(label)
        btnAdd = QPushButton('Add')
        btnAdd.setToolTip(label)
        self.connect(btnSave, SIGNAL("clicked()"), self.saveClicked)
        self.connect(btnAdd, SIGNAL("clicked()"), self.addClicked)
        hbox = QHBoxLayout()
        hbox.addWidget(btnAdd)
        hbox.addWidget(btnSave)
        vbox.addLayout(hbox)
        return vbox, lnEdits

    def saveClicked(self):
        sender_obj = self.sender()
        lbl = str(sender_obj.toolTip())
        lnEdits = self.lbls[lbl]
        newDict = {}
        for ed in lnEdits:
            txt = str(ed.text())
            if '=' in txt:
                key = txt.split('=')[0].strip()
                val = txt.split('=')[1].strip()
                try:
                    val = eval(val)
                except:
                    pass
                newDict[key] = val
        self.parent().parent().parent().saveLbls(lbl, newDict)

    def addClicked(self):
        sender_obj = self.sender()
        lbl = str(sender_obj.toolTip())
        self.parent().parent().parent().addAttr(lbl)

    def repopulate(self, item):
        self.removeRecurs(self.hbox_lbl_props)
        for cls in item:
            self.add_lbl(cls, item[cls])

    def removeRecurs(self, thing):
        widgets = []
        layouts = []
        for i in reversed(range(thing.count())):
            itm = thing.itemAt(i)
            if type(itm) is QVBoxLayout or type(itm) is QHBoxLayout:
                layouts.append(itm)
            else:
                widgets.append(itm.widget()) # layout.itemAt(i).widget()
        #print('widgets:'+str(widgets))
        #print('layouts:'+str(layouts))
        for layout in layouts:
            self.removeRecurs(layout)
            thing.removeItem(layout)
        for widget in widgets:
            widget.setParent(None)

class FormWidget(QWidget):

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent) # super(FormWidget, self).__init__(parent)
        self.__controls()
        self.__layout()

    def __controls(self):
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, 4000, 4000)
        self.scene.setItemIndexMethod(QGraphicsScene.NoIndex)
        #self.scene.setAlignment((pyQt4.AlignLeft | pyQt4.AlignTop))
        self.view = QGraphicsView()
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setScene(self.scene)
        #self.view.centerOn(0, 0)
        self.view.setAlignment(Qt.Alignment(Qt.AlignTop | Qt.AlignLeft))
        self.fileList = QListWidget()
        self.fileList.setSelectionMode(QListWidget.ExtendedSelection)
        self.fileList.setDragDropMode(QListWidget.DragOnly)
        self.repopulateFileWidget([])
        self.roiList = QListWidget()
        self.roiList.setSelectionMode(QListWidget.ExtendedSelection)
        self.roiList.setDragDropMode(QListWidget.DragOnly)

    def __layout(self):
        self.vbox = QVBoxLayout()
        self.split = QSplitter()
        self.split.setOrientation(Qt.Horizontal)
        self.split.addWidget(self.fileList)
        self.split.addWidget(self.roiList)
        self.split.addWidget(self.view)
        self.vbox.addWidget(self.split)
        self.setLayout(self.vbox)
        
    def make_hbox(self, widgets):
        hbox = QHBoxLayout()
        for w in widgets:
            w.setFont(QFont('Arial', 9))
            hbox.addWidget(w)
        return hbox

    def repopulateFileWidget(self, files=None):
        self.fileList.clear()
        for (name, path) in files:
            item = QListWidgetItem(name) # was lbl
            item.setData(Qt.UserRole, path)
            self.fileList.addItem(item)

    def repopulateROIList(self, item):
        self.roiList.clear()
        for roi in item['roi']:
            i = QListWidgetItem(roi)
            i.setData(Qt.UserRole, 42)
            self.roiList.addItem(i)

    def refresh_controls(self):
        pass
        #self.btnBlur.setEnabled('level' in txt)
        # self.btnCanny.setEnabled('level' in txt and color == 'gray')

import json

class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.cvImg = None
        self._zoom = 100.0
        self.roi = None
        self.filename = ""
        self.working_dir = None
        self.dataset = {'path':'','items':{}}
        self.dataset_ground_truth = None
        self.displayGroundTruth = True
        self.face = FaceDRT()
        # --------------------------------------
        self.centralWidget = CentralWidget()
        self.setCentralWidget(self.centralWidget)
        self.form = self.centralWidget.formWidget
        self.labelList = self.centralWidget.labelList
        self.setWindowTitle("BLISS - Biometric Labeller & Image Segmentation Sandbox")
        self.statusBar().showMessage("Status Bar")
        self.pxmapItem = CVImageGraphicsItem()
        self.form.scene.addItem(self.pxmapItem)
        self.createActions()
        self.createMenus()
        self.clipboard = None

    def roiSelected(self, roi):
        i = self.dataset['items'][self.filename.split('/')[-1]]
        i['roi'][str(roi)] = {}
        self.form.repopulateROIList(i)
        self.refreshImg()

    # TODO TODO TODO
    def doubleClickPic(self, pt):
        self.statusBar().showMessage(str(pt))
        self.clipboard.setText(str(pt))
        # find eye object with center that is closest to double click and move it to new position.
        eyes = []
        if len(self.form.fileList.selectedItems()) > 0:
            itm = self.form.fileList.selectedItems()[-1]
            name = str(itm.text())
            f = self.dataset['items'][name]
            for roi_ in f['roi']:
                roi = f['roi'][roi_]
                for lbl_ in roi:
                    lbl = roi[lbl_]
                    if lbl_ in ['eye']:
                        eyes.append((lbl, lbl['center']))
            closest_eye = None
            min_dist = None
            for eye in eyes:
                dist = math.sqrt((eye[1][0]-pt[0])**2 + (eye[1][1]-pt[1])**2)
                if closest_eye is None or dist < min_dist:
                    closest_eye = eye
                    min_dist = dist
            closest_eye[0]['center'] = (pt[0], pt[1])
            self.refreshImg()
        else:
            return
        
    def saveLbls(self, lbl, newDict):
        #print('saveLlbs called! yay! '+str(lbl)+' '+str(newDict))
        itm = self.selectedItem()
        roi = self.selectedROIs()[0]
        if itm and roi:
            itm['roi'][roi][lbl] = newDict
            self.labelList.repopulate(itm['roi'][roi])
            self.refreshImg()

    def addAttr(self, lbl):
        itm = self.selectedItem()
        roi = self.selectedROIs()[0]
        if itm and roi:
            itm['roi'][roi][lbl]['newattr'] = 'val'
            self.labelList.repopulate(itm['roi'][roi])
            self.refreshImg()

    def createActions(self):
        self.loadDirectoryAction = self.createAction('Load Directory')
        self.loadJSONAction = self.createAction('Load JSON')
        self.saveAction = self.createAction('Save As')
        self.exitAction = self.createAction('Quit')
        self.segmentFaceEye = self.createAction('Segment Face-Eyes')
        self.addLabelAction = self.createAction('Add Label')
        self.toggleDisplayGroundTruthAction = self.createAction('Toggle Ground Truth')
        self.zoom200Action = self.createAction('zoom 200%')
        self.zoom100Action = self.createAction('zoom 100%')
        self.zoom90Action = self.createAction('zoom 90%')
        self.zoom80Action = self.createAction('zoom 80%')
        self.zoom70Action = self.createAction('zoom 70%')
        self.zoom60Action = self.createAction('zoom 60%')
        self.zoom50Action = self.createAction('zoom 50%')
        self.zoom40Action = self.createAction('zoom 40%')
        self.zoom30Action = self.createAction('zoom 30%')
        self.zoom20Action = self.createAction('zoom 20%')
        self.zoom10Action = self.createAction('zoom 10%')
        self.labelPersonIDJAMBDCAction = self.createAction('Label PersonID JAMBDC')
        self.labelPersonIDMBGCAction = self.createAction('Label PersonID MBGC')
        self.loadGroundTruthAction = self.createAction('Load Ground Truth')
        self.exportFaceDatasetAction = self.createAction('Export Face Dataset')
        self.exportPeriocularDatasetAction = self.createAction('Export Periocular Dataset')
        self.rotate90ClockwiseAction = self.createAction('Rotate 90 Clockwise')
        self.rotate90CounterclockwiseAction = self.createAction('Rotate 90 Counterclockwise')
        self.resizeCropAction = self.createAction('Resize and Crop')
        self.resizePadAction = self.createAction('Resize and Pad')
        self.groupByClassAction = self.createAction('Group by Class')
        self.ungroupAction = self.createAction('Ungroup')
        self.equalizeSamplesAction = self.createAction('Truncate/Equalize Samples')
        self.partitionSamplesAction = self.createAction('Partition Samples')
        self.unpartitionSamplesAction = self.createAction('Unpartition Samples')
        self.augmentMirrorAction = self.createAction('Mirror')
        self.augmentEqualizeHistAction = self.createAction('Histogram Equalization')
        self.augmentGaussianBlurAction = self.createAction('Gaussian Blur')
        self.connect(self.exportFaceDatasetAction,SIGNAL("triggered()"),self.exportFaceDataset)
        self.connect(self.rotate90ClockwiseAction,SIGNAL("triggered()"),self.rotate90Clockwise)
        self.connect(self.rotate90CounterclockwiseAction,SIGNAL("triggered()"),self.rotate90Counterclockwise)
        self.connect(self.exportPeriocularDatasetAction,SIGNAL("triggered()"),self.exportPeriocularDataset)
        self.connect(self.exitAction,SIGNAL("triggered()"),self.close)
        self.connect(self.saveAction,SIGNAL("triggered()"),self.openInputDialog)
        self.connect(self.loadDirectoryAction,SIGNAL("triggered()"),self.load)
        self.connect(self.loadJSONAction,SIGNAL("triggered()"),self.loadJSON)
        self.connect(self.segmentFaceEye,SIGNAL("triggered()"),self.segment)
        self.connect(self.addLabelAction,SIGNAL("triggered()"),self.addLabel)
        self.connect(self.toggleDisplayGroundTruthAction,SIGNAL("triggered()"),self.toggleDisplayGroundTruth)
        self.connect(self.zoom200Action,SIGNAL("triggered()"),self.zoom200)
        self.connect(self.zoom100Action,SIGNAL("triggered()"),self.zoom100)
        self.connect(self.zoom90Action,SIGNAL("triggered()"),self.zoom90)
        self.connect(self.zoom80Action,SIGNAL("triggered()"),self.zoom80)
        self.connect(self.zoom70Action,SIGNAL("triggered()"),self.zoom70)
        self.connect(self.zoom60Action,SIGNAL("triggered()"),self.zoom60)
        self.connect(self.zoom50Action,SIGNAL("triggered()"),self.zoom50)
        self.connect(self.zoom40Action,SIGNAL("triggered()"),self.zoom40)
        self.connect(self.zoom30Action,SIGNAL("triggered()"),self.zoom30)
        self.connect(self.zoom20Action,SIGNAL("triggered()"),self.zoom20)
        self.connect(self.zoom10Action,SIGNAL("triggered()"),self.zoom10)
        self.connect(self.labelPersonIDJAMBDCAction,SIGNAL("triggered()"),self.labelPersonIDJAMBDC)
        self.connect(self.labelPersonIDMBGCAction,SIGNAL("triggered()"),self.labelPersonIDMBGC)
        self.connect(self.loadGroundTruthAction,SIGNAL("triggered()"),self.loadGroundTruth)
        self.connect(self.resizeCropAction,SIGNAL("triggered()"),self.resizeCrop)
        self.connect(self.resizePadAction,SIGNAL("triggered()"),self.resizePad)
        self.connect(self.groupByClassAction, SIGNAL("triggered()"), self.groupByClass)
        self.connect(self.ungroupAction, SIGNAL("triggered()"), self.ungroup)
        self.connect(self.equalizeSamplesAction, SIGNAL("triggered()"), self.equalizeSamples)
        self.connect(self.partitionSamplesAction, SIGNAL("triggered()"), self.partitionSamples)
        self.connect(self.unpartitionSamplesAction, SIGNAL("triggered()"), self.unpartitionSamples)
        self.connect(self.augmentMirrorAction, SIGNAL("triggered()"), self.augmentMirror)
        self.connect(self.augmentEqualizeHistAction, SIGNAL("triggered()"), self.augmentEqualizeHist)
        self.connect(self.augmentGaussianBlurAction, SIGNAL("triggered()"), self.augmentGaussianBlur)
        # ---------------------------------------------------------------
        self.connect(self.form.fileList, SIGNAL("itemSelectionChanged()"), self.loadSelectedItem)
        self.connect(self.form.roiList, SIGNAL("itemSelectionChanged()"), self.roiChanged)
        shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self.form.roiList)
        self.connect(shortcut, SIGNAL("activated()"), self.deleteSelectedROIs)

    def deleteSelectedROIs(self):
        print('delete rois!')
        selectedROIs = self.selectedROIs()
        i = self.selectedItem()
        new_roi = {}
        for roi in i['roi']:
            if roi not in selectedROIs:
                new_roi[roi] = i['roi'][roi]
        i['roi'] = new_roi
        self.form.repopulateROIList(i)
        self.refreshImg()

    def selectedROIs(self):
        return [str(i.text()) for i in self.form.roiList.selectedItems()]

    def selectedItem(self):
        return self.dataset['items'][self.filename.split('/')[-1]]

    def addLabel(self):
        text, result = QInputDialog.getText(self, "Label", "Provide Label Name")
        lbl = str(text)
        if result and text:
            # add label to selected regions of interest.
            itm = self.selectedItem()
            #print('itm = '+str(itm))
            for roi in self.selectedROIs():
                itm['roi'][roi][lbl] = self.getDefaultAttributes(lbl)
            roi = self.selectedROIs()[0]
            self.labelList.repopulate(itm['roi'][roi])
            self.refreshImg()

    def getDefaultAttributes(self, lbl):
        attributes = {}
        for itm in self.dataset['items']:
            itm_ = self.dataset['items'][itm]
            for roi in itm_['roi']:
                roi_ = itm_['roi'][roi]
                if lbl in roi_:
                    attrs = roi_[lbl]
                    for attr in attrs:
                        attributes[attr] = attrs[attr]
        for attr in attributes:
            val = ''
            try:
                val = eval(attributes[attr])
            except:
                pass
            if type(val) is tuple:
                val = (10, 10)
            elif type(val) is int:
                val = 42
            attributes[attr] = val
        return attributes

    def createAction(self, name, icon=':/fileopen.png', shortcut='Ctrl+N', tip='Load a directory'):
        act=QAction(QIcon(icon),self.tr(name),self)
        act.setShortcut(shortcut)
        act.setStatusTip(self.tr(tip))
        return act
        
    def createMenus(self):
        fileMenu=self.menuBar().addMenu(self.tr("File"))
        fileMenu.addAction(self.loadDirectoryAction)
        fileMenu.addAction(self.loadJSONAction)
        fileMenu.addAction(self.saveAction)
        fileMenu.addAction(self.exitAction)
        exportMenu=self.menuBar().addMenu(self.tr("Export"))     # TODO TODO TODO
        exportMenu.addAction(self.exportPeriocularDatasetAction) #
        exportMenu.addAction(self.exportFaceDatasetAction)
        zoomMenu=self.menuBar().addMenu(self.tr("Zoom"))
        zoomMenu.addAction(self.zoom200Action)
        zoomMenu.addAction(self.zoom100Action)
        zoomMenu.addAction(self.zoom90Action)
        zoomMenu.addAction(self.zoom80Action)
        zoomMenu.addAction(self.zoom70Action)
        zoomMenu.addAction(self.zoom60Action)
        zoomMenu.addAction(self.zoom50Action)
        zoomMenu.addAction(self.zoom40Action)
        zoomMenu.addAction(self.zoom30Action)
        zoomMenu.addAction(self.zoom20Action)
        zoomMenu.addAction(self.zoom10Action)
        transformMenu=self.menuBar().addMenu(self.tr("Transform"))
        transformMenu.addAction(self.rotate90ClockwiseAction)
        transformMenu.addAction(self.rotate90CounterclockwiseAction)
        functionMenu=self.menuBar().addMenu(self.tr("Segment"))
        functionMenu.addAction(self.segmentFaceEye)
        labelMenu=self.menuBar().addMenu(self.tr("Label"))
        labelMenu.addAction(self.addLabelAction)
        testingMenu=self.menuBar().addMenu(self.tr("Test"))
        testingMenu.addAction(self.loadGroundTruthAction)
        displayMenu=self.menuBar().addMenu(self.tr("Display"))
        dataMungingMenu=self.menuBar().addMenu(self.tr("DataMunging"))
        dataMungingMenu.addAction(self.resizeCropAction)
        dataMungingMenu.addAction(self.resizePadAction)
        dataMungingMenu.addAction(self.groupByClassAction)
        dataMungingMenu.addAction(self.ungroupAction)
        dataMungingMenu.addAction(self.equalizeSamplesAction)
        dataMungingMenu.addAction(self.partitionSamplesAction)
        dataMungingMenu.addAction(self.unpartitionSamplesAction)
        augmentMenu=self.menuBar().addMenu(self.tr("Oversample"))
        augmentMenu.addAction(self.augmentMirrorAction)
        augmentMenu.addAction(self.augmentEqualizeHistAction)
        augmentMenu.addAction(self.augmentGaussianBlurAction)
        miscMenu=self.menuBar().addMenu(self.tr("Misc"))
        miscMenu.addAction(self.labelPersonIDJAMBDCAction)
        miscMenu.addAction(self.labelPersonIDMBGCAction)
        displayMenu.addAction(self.toggleDisplayGroundTruthAction)

    # label all faces and eyes with personID extracted from file-name for JAMBDC dataset.
    def labelPersonIDJAMBDC(self):
        for f_ in self.dataset['items']:
            f = self.dataset['items'][f_]
            personID = int(f_.split('_')[0])
            for roi_ in f['roi']:
                roi = f['roi'][roi_]
                for lbl_ in roi:
                    lbl = roi[lbl_]
                    if lbl_ in ['face', 'eye']:
                        lbl['personID'] = personID

    # label all faces and eyes with personID extracted from file-name for JAMBDC dataset.
    def labelPersonIDMBGC(self):
        for f_ in self.dataset['items']:
            f = self.dataset['items'][f_]
            personID = int(f_.split('/')[-1][0:5]) # first 5 characters.
            for roi_ in f['roi']:
                roi = f['roi'][roi_]
                for lbl_ in roi:
                    lbl = roi[lbl_]
                    if lbl_ in ['face', 'eye']:
                        lbl['personID'] = personID

    # prompt user to select ground truth file to be loaded for testing.
    def loadGroundTruth(self):
        f = unicode(QFileDialog.getOpenFileName(self, "Choose", "~", '*.json'))
        if f:
            # verify that f is from the same directory as self.working_dir
            working_dir = '/'+'/'.join(f.split('/')[0:-1])
            if working_dir == self.working_dir:
                with open(f) as data_file:
                    self.dataset_ground_truth = json.load(data_file)
                # TODO validate ground truth dataset to make sure it's compatible.
                self.statusBar().showMessage("Load Ground Truth: Done.")
            else:
                self.statusBar().showMessage("Load Ground Truth: Invalid Directory.")

    def toggleDisplayGroundTruth(self):
        self.displayGroundTruth = not self.displayGroundTruth
        self.refreshImg()

    def openInputDialog(self):
        text, result = QInputDialog.getText(self, "Save As", "Provide file name ending in .json")
        if result:
            self.save(text)

    def load(self):
        self.dataset = {'path':'','items':{}}
        root_dir = unicode(QFileDialog.getExistingDirectory(self, "Choose", "~"))
        self.working_dir = root_dir
        rel_root_dir = '/' + root_dir.split('/')[-1] + '/'
        self.dataset['path'] = rel_root_dir
        imported_file_paths = []
        for full_dir, subFolders, files in os.walk(root_dir):
            for f in files:
                rel_path = os.path.relpath(full_dir, os.path.commonprefix([root_dir, full_dir]))
                imported_file_paths.append(os.path.join(rel_path, f))
        files = self.import_files(imported_file_paths)
        for (name, path) in files:
            self.dataset['items'][name] = {'path':path,'roi':{}}
        self.form.repopulateFileWidget(files)

    def loadJSON(self):
        f = unicode(QFileDialog.getOpenFileName(self, "Choose", "~", '*.json'))
        if f:
            with open(f) as data_file:    
                self.dataset = json.load(data_file)
            self.working_dir = '/'+'/'.join(f.split('/')[0:-1])
            paths = [(i, self.dataset['items'][i]['path']) for i in self.dataset['items']]
            self.form.repopulateFileWidget(paths)

    def save(self, name):
        name_ = name
        if not '.json' in name_:
            name_ = name_ + '.json'
        with open(self.working_dir + '/' + name_, 'w') as fp:
            json.dump(self.dataset, fp, sort_keys=True, indent=4, separators=(',',': '))

    def import_files(self, path_list):
        file_list = []
        if len(path_list) == 0:
            return []
        else:
            for path in path_list:
                parts = path.split("/")
                fname = parts[-1]
                print(path)
                if len(fname.split('.')) > 1 and fname.split('.')[1].lower() in ['jpg','jpeg','png','gif','bmp','tif']:
                    file_list.append((fname, path[1:] if path[0] == '.' else '/'+path))
            return file_list

    def loadSelectedItem(self):
        if len(self.form.fileList.selectedItems()) > 0:
            self.refreshImg()
            itm = self.form.fileList.selectedItems()[-1]
            name = str(itm.text())
            i = self.dataset['items'][name]
            self.form.repopulateROIList(i)

    def roiChanged(self):
        self.refreshImg()
        self.labelList.repopulate(self.selectedItem()['roi'][self.selectedROIs()[0]])

    def refreshImg(self):
        # TODO ALTER THIS. It's ugly.
        itm = self.form.fileList.selectedItems()[-1]
        path = str(itm.data(Qt.UserRole).toString())
        print(self.working_dir + path)
        self.loadFile(self.working_dir + path)

    # Perform face/eye segmentation on selected images.
    # Removes and overwrites any existing face segmentation.
    def segment(self):
        min_size_ = 0.05
        text, result = QInputDialog.getText(self, "Min Size", "Specify min-size as % of img width (0.0 - 1.0): Default 0.05")
        if result:
            try:
                min_size_ = float(text)
            except:
                pass
        else:
            return
        print('min_size_ =', min_size_)
        numImgs = len(self.form.fileList.selectedItems())
        for idx, listItem in enumerate(self.form.fileList.selectedItems()):
            i = str(listItem.text())
            try:
                self.statusBar().showMessage("Processing: "+str(idx+1)+'/'+str(numImgs))
                # Clear out all regions of interest for this data item. TODO temporary sledge hammer.
                self.dataset['items'][i]['roi'] = {}
                path = self.dataset['items'][i]['path']
                path = self.working_dir + path
                print(path)
                img = cv2.imread(path)
                if img is not None:
                    faces = self.face.detect(img, min_size=min_size_, small_width=5000)
                    for face in faces:
                        (x,y,w,h) = face.cascade_box
                        (lrx,lry,lrw,lrh) = face.left_eye.cascade_box
                        (rrx,rry,rrw,rrh) = face.right_eye.cascade_box
                        (lx,ly) = face.left_eye.center
                        (rx,ry) = face.right_eye.center
                        confidence = 1.0
                        self.dataset['items'][i]['roi'][str((x,y,w,h))] = {'face':{'confidence':confidence}}
                        self.dataset['items'][i]['roi'][str((rrx,rry,rrw,rrh))] = {'eye':{'side':'right','center':(rx,ry)}}
                        self.dataset['items'][i]['roi'][str((lrx,lry,lrw,lrh))] = {'eye':{'side':'left','center':(lx,ly)}}
            except:
                print('Error segment. Skipping...')
        self.loadSelectedItem()

    def rotate90Clockwise(self):
        self.rotateImage(degrees=-90)

    def rotate90Counterclockwise(self):
        self.rotateImage(degrees=90)

    def rotateImage(self, degrees=90):
        itm = self.form.fileList.selectedItems()[-1]
        path = str(itm.data(Qt.UserRole).toString())
        path_ = self.working_dir + path
        img = cv2.imread(path_)
        rows0, cols0, _ = img.shape
        rows, cols, _ = img.shape
        if rows > cols:
            img = dmu._resize_pad(img, target_width=rows)
            cols = rows
        else:
            img = dmu._resize_pad(img, target_width=cols)
            rows = cols
        M = cv2.getRotationMatrix2D((cols/2,rows/2),degrees,1)
        dst = cv2.warpAffine(img,M,(cols,rows))
        if rows0 > cols0:
            insert_col = (cols-cols0)/2
            dst = dst[insert_col:insert_col+cols0,0:rows0]
        else:
            insert_row = (rows-rows0)/2
            dst = dst[0:cols0,insert_row:insert_row+rows0]
        cv2.imwrite(path_, dst)
        self.refreshImg()

    # TODO This uses too much RAM. Please rewrite. TODO
    # TODO Add minimum source resolution parameter so we can ignore crops with too few pixels.
    # TODO Add output resolution parameter so faces can be something other than 512 x 512
    def _exportFaceDataset(self, eye=False):
        print('Export Dataset')
        crop_size = 0.23
        text, result = QInputDialog.getText(self, "Crop Size", "Specify face crop size. Default "+str(crop_size))
        if result:
            try:
                crop_size = float(text)
            except:
                pass
        else:
            return
        min_src_width = 64 # minimum distance between eyes in pixels within source image.
        msg = "Specify minimum distance between eyes in pixels. Default "+str(min_src_width)
        text, result = QInputDialog.getText(self, "Minimum Eye Dist", msg)
        if result:
            try:
                min_src_width = int(text)
            except:
                pass
        else:
            return
        # -------------------------
        folder_name = self.dataset['path'].split('/')[-2]
        path_ = '/'.join(self.working_dir.split('/')[0:-1])
        if eye:
            path_ = path_+'/'+folder_name+'_periocular'
        else:
            path_ = path_+'/'+folder_name+'_faces'
        os.mkdir(path_)
        # -------------------------
        numImgs = len(self.form.fileList.selectedItems())
        for idx, listItem in enumerate(self.form.fileList.selectedItems()):
            i = str(listItem.text())
            self.statusBar().showMessage("Processing: "+str(idx+1)+'/'+str(numImgs))
            path = self.dataset['items'][i]['path']
            path = self.working_dir + path
            print(path)
            #face_roi = None
            #right_eye_roi = None
            #left_eye_roi = None
            left_eye_center = None
            right_eye_center = None
            person_id = None
            f = self.dataset['items'][i]
            file_name = f['path']
            for roi_ in f['roi']:
                roi = f['roi'][roi_]
                for lbl_ in roi:
                    lbl = roi[lbl_]
                    if lbl_ == 'eye':
                        if lbl['side'] == 'right':
                            right_eye_center = lbl['center']
                        else:
                            left_eye_center = lbl['center']
            if left_eye_center is not None and right_eye_center is not None:
                dist = int(math.sqrt((left_eye_center[0]-right_eye_center[0])**2.0 + (left_eye_center[1]-right_eye_center[1])**2.0))
                if dist >= min_src_width:
                    img = cv2.imread(path)
                    if img is not None:
                        face = Face()
                        cropped = face.cropFace(img, left_eye_center, right_eye_center, (crop_size,crop_size), (512,512))
                        if eye:
                            shift = int((crop_size - 0.23)*100)*7
                            left_eye = cropped[0+shift:220+shift,0+shift:220+shift]     # Assumes 512x512 TODO
                            right_eye = cropped[0+shift:220+shift,292-shift:512-shift]
                            exten = file_name.split('.')[-1]
                            name = file_name.split('.')[-2]
                            cv2.imwrite(path_+name+'_left.'+exten, left_eye)
                            cv2.imwrite(path_+name+'_right.'+exten, right_eye)
                        else:
                            cv2.imwrite(path_+file_name, cropped)
                else:
                    print('Source image too small. Skipping.')

    def exportPeriocularDataset(self):
        self._exportFaceDataset(eye=True)

    def exportFaceDataset(self):
        self._exportFaceDataset(eye=False)

    def resizeCrop(self):
        _width = 224
        text, result = QInputDialog.getText(self, "width", "specify width in pixels: default 224")
        if result:
            try:
                _width = int(text)
            except:
                pass
        else:
            return # user clicked "cancelled".
        if _width > 0 and _width <= 1000:
            self.statusBar().showMessage("resizing and cropping...")
            dmu.resize_crop(path=self.working_dir, target_width=_width)
            self.statusBar().showMessage("done resizing and cropping")
        else:
            self.statusBar().showMessage("width beyond limits.")

    def resizePad(self):
        _width = 224
        text, result = QInputDialog.getText(self, "width", "specify width in pixels: default 224")
        if result:
            try:
                _width = int(text)
            except:
                pass
        else:
            return # user clicked "cancelled".
        if _width > 0 and _width <= 1000:
            self.statusBar().showMessage("resizing and padding...")
            dmu.resize_pad(path=self.working_dir, target_width=_width)
            self.statusBar().showMessage("done resizing and padding")
        else:
            self.statusBar().showMessage("width beyond limits.")

    def groupByClass(self):
        delimiters = [' ', '.', '-', '_']
        text, result = QInputDialog.getText(self, "delimiter", "specify delimiter if something other than a blank space")
        if result:
            try:
                delimiters = [text]
            except:
                pass
        else:
            return # user clicked "cancelled".
        self.statusBar().showMessage("grouping by class...")
        dmu.group_into_class_folders(path=self.working_dir, delimiters=delimiters)
        self.statusBar().showMessage("done grouping by class")

    def ungroup(self):
        self.statusBar().showMessage("ungrouping...")
        dmu.ungroup(path=self.working_dir)
        self.statusBar().showMessage("done ungrouping")

    def equalizeSamples(self):
        n = 224
        text, result = QInputDialog.getText(self, "samples", "how many samples per class?")
        if result:
            try:
                n = int(text)
            except:
                pass
        else:
            return # user clicked "cancelled".
        if n > 0:
            self.statusBar().showMessage("equalizing number of samples...")
            dmu.equalize_num_samples(path=self.working_dir, n=n)
            self.statusBar().showMessage("done equalizing number of samples")
        else:
            self.statusBar().showMessage("samples beyond limits.")

    def partitionSamples(self):
        train, val, test = -1, -1, -1
        text, result = QInputDialog.getText(self, "training, validation, testing", "Enter number of training, validation, and testing samples as ints separated by spaces.")
        if result:
            try:
                parts = text.split(' ')
                train, val, test = int(parts[0]), int(parts[1]), int(parts[2])
            except:
                self.statusBar().showMessage("invalid")
                return
        else:
            return # user clicked "cancelled".
        self.statusBar().showMessage("partitioning samples...")
        dmu.partition(path=self.working_dir, training=train, validation=val, testing=test)
        self.statusBar().showMessage("done partitioning samples")

    def unpartitionSamples(self):
        self.statusBar().showMessage("unpartitioning samples...")
        dmu.unpartition(path=self.working_dir)
        self.statusBar().showMessage("done unpartitioning samples")

    def augmentMirror(self):
        self.statusBar().showMessage("mirror...")
        dmu.augment(path=self.working_dir, percent_increase=1.0, augmentations=['mirror'])
        self.statusBar().showMessage("done creating oversampled set.")

    def augmentEqualizeHist(self):
        self.statusBar().showMessage("equalize hist...")
        dmu.augment(path=self.working_dir, percent_increase=1.0, augmentations=['equalize_hist'])
        self.statusBar().showMessage("done equalizing hist.")

    def augmentGaussianBlur(self):
        self.statusBar().showMessage("gaussian blur...")
        dmu.augment(path=self.working_dir, percent_increase=1.0, augmentations=['gaussian_blur'])
        self.statusBar().showMessage("done with gaussian blur.")

    def zoom(self, value):
        self._zoom = value
        self.statusBar().showMessage("Zoom "+str(self._zoom)+"%")
        factor = value / 100.0
        matrix = self.form.view.matrix()
        matrix.reset()
        matrix.scale(factor, factor)
        self.form.view.setMatrix(matrix)

    def zoom_(self, value):
        self._zoom = value
        self.loadFile(self.filename)
        self.zoom(value)

    def zoom200(self):
        self.zoom_(200.0)
    def zoom100(self):
        self.zoom_(100.0)
    def zoom90(self):
        self.zoom_(90.0)
    def zoom80(self):
        self.zoom_(80.0)
    def zoom70(self):
        self.zoom_(70.0)
    def zoom60(self):
        self.zoom_(60.0)
    def zoom50(self):
        self.zoom_(50.0)
    def zoom40(self):
        self.zoom_(40.0)
    def zoom30(self):
        self.zoom_(30.0)
    def zoom20(self):
        self.zoom_(20.0)
    def zoom10(self):
        self.zoom_(10.0)

    def loadFile(self, fname=None):
        thickness = self.getLineThickness()
        if fname: # and fname != self.filename:
            self.filename = None
            self.filename = fname
            self.cvImg = cv2.imread(fname) #read image with opencv
            name = fname.split('/')[-1]
            meta = self.dataset['items'][name]
            selectedROIs = [str(i.text()) for i in self.form.roiList.selectedItems()]
            #print('selectedROIs =' + str(selectedROIs))
            if self.dataset_ground_truth is not None and self.displayGroundTruth:
                meta_t = self.dataset_ground_truth['items'][name]
                for roi in meta_t['roi']:
                    (x,y,w,h) = eval(roi)
                    cv2.rectangle(self.cvImg,(x,y),(x+w,y+h),(255,0,255),thickness)
                    d = meta_t['roi'][roi]
                    for class_ in d:
                        for attribute in d[class_]:
                            value = d[class_][attribute]
                            try:
                                tpl = tuple(value)
                                if len(tpl) == 2:
                                    cv2.circle(self.cvImg, tpl, 5, (255,0,255), thickness)
                            except:
                                pass
            for roi in meta['roi']:
                (x,y,w,h) = eval(roi)
                x, y, w, h = int(x), int(y), int(w), int(h)
                if roi in selectedROIs:
                    cv2.rectangle(self.cvImg,(x,y),(x+w,y+h),(150,150,0),thickness)
                else:
                    #print('x, y, w, h =', x, y, w, h)
                    cv2.rectangle(self.cvImg,(x,y),(x+w,y+h),(255,0,0),thickness)
                d = meta['roi'][roi]
                for class_ in d:
                    # TODO draw class name as text label.
                    for attribute in d[class_]:
                        value = d[class_][attribute]
                        # if the value is a tuple or list of length two integers, then draw circles.
                        try:
                            tpl = tuple(value)
                            if len(tpl) == 2: # TODO Need to verify integer type.
                                cv2.circle(self.cvImg, tpl, 5, (255,255,0), thickness)
                        except:
                            pass
            #self.imageBinder.load(fname.split('/')[-1])
            self.updateImg()
            self.zoom(self._zoom)

    def getLineThickness(self):
        return max(2, int(2.0 * (100.0/self._zoom)))

    def updateImg(self):
        if self.cvImg != None:
            self.roi = None
            self.pxmapItem.updateImg(self.cvImg)
            self.form.view.centerOn(0,0)
            #self.form.scene.clear()
            #self.form.scene.addItem(self.pxmapItem)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('./icons/wing_icon.jpg'))
    form = MainWindow()
    form.clipboard = app.clipboard()
    form.resize(800, 600)
    form.show()
    app.exec_()
