# PyQt5 pour l'interface graphique et ses composants
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

# Numpy pour la manipulation des matrices et tableaux
import numpy as np

# skimage est une librairie de traitement d'image (binarisation , lecture , ... )
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

# Scikit-learn librairie de machine learning
from sklearn import tree
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neural_network import MLPClassifier


# le temps
import time

# Matplotlib pour le traitement des courbes
import matplotlib.pyplot as plt

# la fonction de pre-traitement des images  : importation - conversion - binarisation - aplatissement


def pre_traitement(choix, number):

    # tableaux des images
    images = []
    for i in range(1, number+1):

        # le path d'une image d'indice i
        img = "D:/UCareer/SIR_2020/S2/TAAD/TP/RESSOURCES/images/" + \
            choix.__add__(i.__str__()).__add__(".png")

        # lecture de l'image
        imageRGB = imread(img)

        # conversion à une image au niveau de gris
        image = rgb2gray(imageRGB)

        # récupération de seuil de binarisation basée sur l'algorithme outsu de calcul de seuil de binarisation globale
        thresh = threshold_otsu(image)

        # binarisation de l'image suivant le seuil thresh
        binary = image > thresh

        # l'empilation de l'image dans le tableau et l'Aplatissement dans un tableau monodimensionnel
        images.append(np.ravel(binary))

    return images

# la fonction de l'étiquettage


def d_function(choix):

    # l'étiquetage des classes de chaque image d'apprentissage
    if(choix == 1):
        return np.array([
                        1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7,
                        7, 8, 8, 9, 9, 9, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
                        14, 14, 15, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20,
                        20, 21, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26]
                        )

    # l'étiquetage des classes de chaque image de test
    else:
        return np.array([
                        1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 12, 13, 13,
                        14, 14, 15, 15, 16, 16, 17, 18, 19, 19, 20, 21, 21, 22, 23, 24, 25, 26]
                        )


# fonction d'apprentissage Decision Tree classifier

def learning_DecisionTreeClassifier(images, d):

    # t0 pour compter le temps d'execution
    t0 = time.perf_counter()
    cl = tree.DecisionTreeClassifier(criterion='entropy')
    cl.fit(images, d)
    t1 = time.perf_counter()
    t2 = t1-t0
    return cl, t2

# fonction d'apprentissage Extra Tree classifier


def learning_ExtraTreeClassifier(images, d):

    # t0 pour compter le temps d'execution
    t3 = time.perf_counter()
    cl = tree.ExtraTreeClassifier(random_state=0, criterion='entropy')
    cl.fit(images, d)
    t4 = time.perf_counter()
    t5 = t4-t3
    return cl, t5


# fonction d'apprentissage gaussien

def learning_gaussien(images, d):

    # t0 pour compter le temps d'execution
    t0 = time.perf_counter()
    gaussien = GaussianNB(var_smoothing=0.9)
    gaussien.fit(images, d)
    t1 = time.perf_counter()
    t2 = t1-t0
    return gaussien, t2

# fonction d'apprentissage bernoulli


def learning_bernoulli(images, d):

    # t0 pour compter le temps d'execution
    t3 = time.perf_counter()
    bernoulli = BernoulliNB()
    bernoulli.fit(images, d)
    t4 = time.perf_counter()
    t5 = t4-t3
    return bernoulli, t5

# fonction d'apprentissage multinomial


def learning_multinomial(images, d):

    # t0 pour compter le temps d'execution
    t6 = time.perf_counter()
    multinomial = MultinomialNB()
    multinomial.fit(images, d)
    t7 = time.perf_counter()
    t8 = t7-t6
    return multinomial, t8


# fonction d'apprentissage MLP : perceptron multicouche

def learning_algorithme(images, d):

    t9 = time.perf_counter()
    cl = MLPClassifier(activation="logistic",
                       max_iter=1000,
                       hidden_layer_sizes=200,
                       learning_rate='adaptive'
                       )
    cl.fit(images, d)
    t10 = time.perf_counter()
    t11 = t10 - t9
    return cl, t11

# la fonction de taux de reconnaissance ou de test


def taux(images, reseauN, etiquettage):

    # taux d'apprentissage ou de Test
    taux_reussite = 0
    number = len(images)
    for i in range(0, number):

        # Prédiction
        y_pred = reseauN.predict([images[i]])

        # la classe de l'image de test
        y_pred = np.int(y_pred[0])

        # comparaison des classes
        if y_pred == etiquettage[i]:
            taux_reussite = taux_reussite + 1

    # taux de reussite
    taux_reussite = (taux_reussite/(number))*100

    return taux_reussite


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        # les alphabets des images
        self.alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                         "U", "V", "W", "X", "Y", "Z"]
        # l'image de test apres traitement
        self.imageTest = []

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.index = QtWidgets.QTabWidget(self.centralwidget)
        self.index.setGeometry(QtCore.QRect(30, 20, 741, 521))
        self.index.setObjectName("index")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.bernoulli_pushButton = QtWidgets.QPushButton(self.tab)
        self.bernoulli_pushButton.setGeometry(QtCore.QRect(20, 380, 75, 23))
        self.bernoulli_pushButton.setObjectName("bernoulli_pushButton")
        self.gaussien_pushButton = QtWidgets.QPushButton(self.tab)
        self.gaussien_pushButton.setGeometry(QtCore.QRect(110, 380, 75, 23))
        self.gaussien_pushButton.setObjectName("gaussien_pushButton")
        self.multinomial_pushButton = QtWidgets.QPushButton(self.tab)
        self.multinomial_pushButton.setGeometry(QtCore.QRect(200, 380, 75, 23))
        self.multinomial_pushButton.setObjectName("multinomial_pushButton")
        self.image_test_label = QtWidgets.QLabel(self.tab)
        self.image_test_label.setGeometry(QtCore.QRect(190, 60, 331, 241))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.image_test_label.setFont(font)
        self.image_test_label.setText("")
        self.image_test_label.setObjectName("image_test_label")
        self.field_test_bernoulli = QtWidgets.QLabel(self.tab)
        self.field_test_bernoulli.setGeometry(QtCore.QRect(20, 410, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.field_test_bernoulli.setFont(font)
        self.field_test_bernoulli.setText("")
        self.field_test_bernoulli.setIndent(26)
        self.field_test_bernoulli.setObjectName("field_test_bernoulli")
        self.field_test_gaussien = QtWidgets.QLabel(self.tab)
        self.field_test_gaussien.setGeometry(QtCore.QRect(110, 410, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.field_test_gaussien.setFont(font)
        self.field_test_gaussien.setText("")
        self.field_test_gaussien.setIndent(26)
        self.field_test_gaussien.setObjectName("field_test_gaussien")
        self.field_test_Multinomial = QtWidgets.QLabel(self.tab)
        self.field_test_Multinomial.setGeometry(QtCore.QRect(200, 410, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.field_test_Multinomial.setFont(font)
        self.field_test_Multinomial.setText("")
        self.field_test_Multinomial.setIndent(26)
        self.field_test_Multinomial.setObjectName("field_test_Multinomial")
        self.importButton = QtWidgets.QPushButton(self.tab)
        self.importButton.setGeometry(QtCore.QRect(30, 170, 75, 23))
        self.importButton.setObjectName("importButton")
        self.copyright = QtWidgets.QLabel(self.tab)
        self.copyright.setGeometry(QtCore.QRect(640, 460, 81, 16))
        self.copyright.setObjectName("copyright")
        self.mlp_pushButton = QtWidgets.QPushButton(self.tab)
        self.mlp_pushButton.setGeometry(QtCore.QRect(290, 380, 75, 23))
        self.mlp_pushButton.setObjectName("mlp_pushButton")
        self.field_test_mlp = QtWidgets.QLabel(self.tab)
        self.field_test_mlp.setGeometry(QtCore.QRect(290, 410, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.field_test_mlp.setFont(font)
        self.field_test_mlp.setText("")
        self.field_test_mlp.setIndent(26)
        self.field_test_mlp.setObjectName("field_test_mlp")
        self.field_test_Decisiontreeclassifier = QtWidgets.QLabel(self.tab)
        self.field_test_Decisiontreeclassifier.setGeometry(
            QtCore.QRect(400, 410, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.field_test_Decisiontreeclassifier.setFont(font)
        self.field_test_Decisiontreeclassifier.setText("")
        self.field_test_Decisiontreeclassifier.setIndent(26)
        self.field_test_Decisiontreeclassifier.setObjectName(
            "field_test_Decisiontreeclassifier")
        self.field_test_Extratreeclassifier = QtWidgets.QLabel(self.tab)
        self.field_test_Extratreeclassifier.setGeometry(
            QtCore.QRect(550, 410, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.field_test_Extratreeclassifier.setFont(font)
        self.field_test_Extratreeclassifier.setText("")
        self.field_test_Extratreeclassifier.setIndent(26)
        self.field_test_Extratreeclassifier.setObjectName(
            "field_test_Extratreeclassifier")
        self.Extratreeclassifier_pushButton = QtWidgets.QPushButton(self.tab)
        self.Extratreeclassifier_pushButton.setGeometry(
            QtCore.QRect(520, 380, 131, 23))
        self.Extratreeclassifier_pushButton.setObjectName(
            "Extratreeclassifier_pushButton")
        self.Decisiontreeclassifier_pushButton = QtWidgets.QPushButton(
            self.tab)
        self.Decisiontreeclassifier_pushButton.setGeometry(
            QtCore.QRect(374, 380, 131, 23))
        self.Decisiontreeclassifier_pushButton.setObjectName(
            "Decisiontreeclassifier_pushButton")
        self.index.addTab(self.tab, "")
        self.Gaussien_Tab = QtWidgets.QWidget()
        self.Gaussien_Tab.setObjectName("Gaussien_Tab")
        self.temps_execution_MLP = QtWidgets.QLabel(self.Gaussien_Tab)
        self.temps_execution_MLP.setGeometry(QtCore.QRect(370, 80, 121, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.temps_execution_MLP.setFont(font)
        self.temps_execution_MLP.setStyleSheet("border : 4px solid #000")
        self.temps_execution_MLP.setText("")
        self.temps_execution_MLP.setIndent(49)
        self.temps_execution_MLP.setObjectName("temps_execution_MLP")
        self.taux_test_MLP = QtWidgets.QLabel(self.Gaussien_Tab)
        self.taux_test_MLP.setGeometry(QtCore.QRect(490, 80, 231, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.taux_test_MLP.setFont(font)
        self.taux_test_MLP.setStyleSheet("border : 4px solid #000")
        self.taux_test_MLP.setText("")
        self.taux_test_MLP.setIndent(104)
        self.taux_test_MLP.setObjectName("taux_test_MLP")
        self.taux_reconnaissance_MLP = QtWidgets.QLabel(self.Gaussien_Tab)
        self.taux_reconnaissance_MLP.setGeometry(QtCore.QRect(90, 80, 281, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.taux_reconnaissance_MLP.setFont(font)
        self.taux_reconnaissance_MLP.setStyleSheet("border : 4px solid #000")
        self.taux_reconnaissance_MLP.setText("")
        self.taux_reconnaissance_MLP.setIndent(128)
        self.taux_reconnaissance_MLP.setObjectName("taux_reconnaissance_MLP")
        self.const_label3 = QtWidgets.QLabel(self.Gaussien_Tab)
        self.const_label3.setGeometry(QtCore.QRect(490, 30, 231, 41))
        self.const_label3.setStyleSheet("border : 4px solid #000")
        self.const_label3.setObjectName("const_label3")
        self.const_label1 = QtWidgets.QLabel(self.Gaussien_Tab)
        self.const_label1.setGeometry(QtCore.QRect(90, 30, 281, 41))
        self.const_label1.setStyleSheet("border : 4px solid #000")
        self.const_label1.setObjectName("const_label1")
        self.const_label2 = QtWidgets.QLabel(self.Gaussien_Tab)
        self.const_label2.setGeometry(QtCore.QRect(370, 30, 121, 41))
        self.const_label2.setStyleSheet("border : 4px solid #000")
        self.const_label2.setObjectName("const_label2")
        self.afficherButton = QtWidgets.QPushButton(self.Gaussien_Tab)
        self.afficherButton.setGeometry(QtCore.QRect(340, 460, 75, 23))
        self.afficherButton.setObjectName("afficherButton")
        self.const_label2_2 = QtWidgets.QLabel(self.Gaussien_Tab)
        self.const_label2_2.setGeometry(QtCore.QRect(20, 30, 71, 41))
        self.const_label2_2.setStyleSheet("border : 4px solid #000")
        self.const_label2_2.setObjectName("const_label2_2")
        self.MLP_label = QtWidgets.QLabel(self.Gaussien_Tab)
        self.MLP_label.setGeometry(QtCore.QRect(20, 80, 71, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.MLP_label.setFont(font)
        self.MLP_label.setStyleSheet("border : 4px solid #000")
        self.MLP_label.setIndent(12)
        self.MLP_label.setObjectName("MLP_label")
        self.BERN_label = QtWidgets.QLabel(self.Gaussien_Tab)
        self.BERN_label.setGeometry(QtCore.QRect(20, 140, 71, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.BERN_label.setFont(font)
        self.BERN_label.setStyleSheet("border : 4px solid #000")
        self.BERN_label.setIndent(7)
        self.BERN_label.setObjectName("BERN_label")
        self.taux_test_BERN = QtWidgets.QLabel(self.Gaussien_Tab)
        self.taux_test_BERN.setGeometry(QtCore.QRect(490, 140, 231, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.taux_test_BERN.setFont(font)
        self.taux_test_BERN.setStyleSheet("border : 4px solid #000")
        self.taux_test_BERN.setText("")
        self.taux_test_BERN.setIndent(104)
        self.taux_test_BERN.setObjectName("taux_test_BERN")
        self.temps_execution_BERN = QtWidgets.QLabel(self.Gaussien_Tab)
        self.temps_execution_BERN.setGeometry(QtCore.QRect(370, 140, 121, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.temps_execution_BERN.setFont(font)
        self.temps_execution_BERN.setStyleSheet("border : 4px solid #000")
        self.temps_execution_BERN.setText("")
        self.temps_execution_BERN.setIndent(49)
        self.temps_execution_BERN.setObjectName("temps_execution_BERN")
        self.taux_reconnaissance_BERN = QtWidgets.QLabel(self.Gaussien_Tab)
        self.taux_reconnaissance_BERN.setGeometry(
            QtCore.QRect(90, 140, 281, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.taux_reconnaissance_BERN.setFont(font)
        self.taux_reconnaissance_BERN.setStyleSheet("border : 4px solid #000")
        self.taux_reconnaissance_BERN.setText("")
        self.taux_reconnaissance_BERN.setIndent(128)
        self.taux_reconnaissance_BERN.setObjectName("taux_reconnaissance_BERN")
        self.taux_test_MULTI = QtWidgets.QLabel(self.Gaussien_Tab)
        self.taux_test_MULTI.setGeometry(QtCore.QRect(490, 200, 231, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.taux_test_MULTI.setFont(font)
        self.taux_test_MULTI.setStyleSheet("border : 4px solid #000")
        self.taux_test_MULTI.setText("")
        self.taux_test_MULTI.setIndent(104)
        self.taux_test_MULTI.setObjectName("taux_test_MULTI")
        self.MULTI_label = QtWidgets.QLabel(self.Gaussien_Tab)
        self.MULTI_label.setGeometry(QtCore.QRect(20, 200, 71, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.MULTI_label.setFont(font)
        self.MULTI_label.setStyleSheet("border : 4px solid #000")
        self.MULTI_label.setIndent(2)
        self.MULTI_label.setObjectName("MULTI_label")
        self.taux_reconnaissance_MULTI = QtWidgets.QLabel(self.Gaussien_Tab)
        self.taux_reconnaissance_MULTI.setGeometry(
            QtCore.QRect(90, 200, 281, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.taux_reconnaissance_MULTI.setFont(font)
        self.taux_reconnaissance_MULTI.setStyleSheet("border : 4px solid #000")
        self.taux_reconnaissance_MULTI.setText("")
        self.taux_reconnaissance_MULTI.setIndent(128)
        self.taux_reconnaissance_MULTI.setObjectName(
            "taux_reconnaissance_MULTI")
        self.temps_execution_MULTI = QtWidgets.QLabel(self.Gaussien_Tab)
        self.temps_execution_MULTI.setGeometry(QtCore.QRect(370, 200, 121, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.temps_execution_MULTI.setFont(font)
        self.temps_execution_MULTI.setStyleSheet("border : 4px solid #000")
        self.temps_execution_MULTI.setText("")
        self.temps_execution_MULTI.setIndent(49)
        self.temps_execution_MULTI.setObjectName("temps_execution_MULTI")
        self.BINO_label = QtWidgets.QLabel(self.Gaussien_Tab)
        self.BINO_label.setGeometry(QtCore.QRect(20, 260, 71, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.BINO_label.setFont(font)
        self.BINO_label.setStyleSheet("border : 4px solid #000")
        self.BINO_label.setIndent(6)
        self.BINO_label.setObjectName("BINO_label")
        self.taux_test_GAUS = QtWidgets.QLabel(self.Gaussien_Tab)
        self.taux_test_GAUS.setGeometry(QtCore.QRect(490, 260, 231, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.taux_test_GAUS.setFont(font)
        self.taux_test_GAUS.setStyleSheet("border : 4px solid #000")
        self.taux_test_GAUS.setText("")
        self.taux_test_GAUS.setIndent(104)
        self.taux_test_GAUS.setObjectName("taux_test_GAUS")
        self.temps_execution_GAUSS = QtWidgets.QLabel(self.Gaussien_Tab)
        self.temps_execution_GAUSS.setGeometry(QtCore.QRect(370, 260, 121, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.temps_execution_GAUSS.setFont(font)
        self.temps_execution_GAUSS.setStyleSheet("border : 4px solid #000")
        self.temps_execution_GAUSS.setText("")
        self.temps_execution_GAUSS.setIndent(49)
        self.temps_execution_GAUSS.setObjectName("temps_execution_GAUSS")
        self.taux_reconnaissance_GAUSS = QtWidgets.QLabel(self.Gaussien_Tab)
        self.taux_reconnaissance_GAUSS.setGeometry(
            QtCore.QRect(90, 260, 281, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.taux_reconnaissance_GAUSS.setFont(font)
        self.taux_reconnaissance_GAUSS.setStyleSheet("border : 4px solid #000")
        self.taux_reconnaissance_GAUSS.setText("")
        self.taux_reconnaissance_GAUSS.setIndent(128)
        self.taux_reconnaissance_GAUSS.setObjectName(
            "taux_reconnaissance_GAUSS")
        self.DECT_label = QtWidgets.QLabel(self.Gaussien_Tab)
        self.DECT_label.setGeometry(QtCore.QRect(20, 320, 71, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.DECT_label.setFont(font)
        self.DECT_label.setStyleSheet("border : 4px solid #000")
        self.DECT_label.setIndent(6)
        self.DECT_label.setObjectName("DECT_label")
        self.temps_execution_DECT = QtWidgets.QLabel(self.Gaussien_Tab)
        self.temps_execution_DECT.setGeometry(QtCore.QRect(370, 320, 121, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.temps_execution_DECT.setFont(font)
        self.temps_execution_DECT.setStyleSheet("border : 4px solid #000")
        self.temps_execution_DECT.setText("")
        self.temps_execution_DECT.setIndent(49)
        self.temps_execution_DECT.setObjectName("temps_execution_DECT")
        self.taux_reconnaissance_DECT = QtWidgets.QLabel(self.Gaussien_Tab)
        self.taux_reconnaissance_DECT.setGeometry(
            QtCore.QRect(90, 320, 281, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.taux_reconnaissance_DECT.setFont(font)
        self.taux_reconnaissance_DECT.setStyleSheet("border : 4px solid #000")
        self.taux_reconnaissance_DECT.setText("")
        self.taux_reconnaissance_DECT.setIndent(128)
        self.taux_reconnaissance_DECT.setObjectName("taux_reconnaissance_DECT")
        self.taux_test_DECT = QtWidgets.QLabel(self.Gaussien_Tab)
        self.taux_test_DECT.setGeometry(QtCore.QRect(490, 320, 231, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.taux_test_DECT.setFont(font)
        self.taux_test_DECT.setStyleSheet("border : 4px solid #000")
        self.taux_test_DECT.setText("")
        self.taux_test_DECT.setIndent(104)
        self.taux_test_DECT.setObjectName("taux_test_DECT")
        self.EXTT_label = QtWidgets.QLabel(self.Gaussien_Tab)
        self.EXTT_label.setGeometry(QtCore.QRect(20, 380, 71, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.EXTT_label.setFont(font)
        self.EXTT_label.setStyleSheet("border : 4px solid #000")
        self.EXTT_label.setIndent(7)
        self.EXTT_label.setObjectName("EXTT_label")
        self.temps_execution_EXTT = QtWidgets.QLabel(self.Gaussien_Tab)
        self.temps_execution_EXTT.setGeometry(QtCore.QRect(370, 380, 121, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.temps_execution_EXTT.setFont(font)
        self.temps_execution_EXTT.setStyleSheet("border : 4px solid #000")
        self.temps_execution_EXTT.setText("")
        self.temps_execution_EXTT.setIndent(49)
        self.temps_execution_EXTT.setObjectName("temps_execution_EXTT")
        self.taux_reconnaissance_EXTT = QtWidgets.QLabel(self.Gaussien_Tab)
        self.taux_reconnaissance_EXTT.setGeometry(
            QtCore.QRect(90, 380, 281, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.taux_reconnaissance_EXTT.setFont(font)
        self.taux_reconnaissance_EXTT.setStyleSheet("border : 4px solid #000")
        self.taux_reconnaissance_EXTT.setText("")
        self.taux_reconnaissance_EXTT.setIndent(128)
        self.taux_reconnaissance_EXTT.setObjectName("taux_reconnaissance_EXTT")
        self.taux_test_EXTT = QtWidgets.QLabel(self.Gaussien_Tab)
        self.taux_test_EXTT.setGeometry(QtCore.QRect(490, 380, 231, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.taux_test_EXTT.setFont(font)
        self.taux_test_EXTT.setStyleSheet("border : 4px solid #000")
        self.taux_test_EXTT.setText("")
        self.taux_test_EXTT.setIndent(104)
        self.taux_test_EXTT.setObjectName("taux_test_EXTT")
        self.index.addTab(self.Gaussien_Tab, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.index.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.bernoulli_pushButton.setText(
            _translate("MainWindow", "Bernoulli"))
        self.gaussien_pushButton.setText(_translate("MainWindow", "Gaussien"))
        self.multinomial_pushButton.setText(
            _translate("MainWindow", "Multinomial"))
        self.importButton.setText(_translate("MainWindow", "import"))
        self.copyright.setText(_translate("MainWindow", "Hamza Dabaghi"))
        self.mlp_pushButton.setText(_translate("MainWindow", "MLP"))
        self.Extratreeclassifier_pushButton.setText(
            _translate("MainWindow", "Extratreeclassifier"))
        self.Decisiontreeclassifier_pushButton.setText(
            _translate("MainWindow", "Decisiontreeclassifier"))
        self.index.setTabText(self.index.indexOf(
            self.tab), _translate("MainWindow", "Index"))
        self.const_label3.setText(_translate(
            "MainWindow", "Taux de Reconnaissance des images de Tests"))
        self.const_label1.setText(_translate(
            "MainWindow", "Taux de Reconnaissance des images de l\'apprentissage"))
        self.const_label2.setText(_translate(
            "MainWindow", " Temps d\'execution"))
        self.afficherButton.setText(_translate("MainWindow", "Afficher"))
        self.const_label2_2.setText(_translate("MainWindow", "    Modeles"))
        self.MLP_label.setText(_translate("MainWindow", "MLP"))
        self.BERN_label.setText(_translate("MainWindow", "BERN"))
        self.MULTI_label.setText(_translate("MainWindow", "MULTIN"))
        self.BINO_label.setText(_translate("MainWindow", "GAUS"))
        self.DECT_label.setText(_translate("MainWindow", "DECT"))
        self.EXTT_label.setText(_translate("MainWindow", "EXTT"))
        self.index.setTabText(self.index.indexOf(
            self.Gaussien_Tab), _translate("MainWindow", "Modeles"))

        self.importButton.clicked.connect(self.openFile)
        self.Decisiontreeclassifier_pushButton.clicked.connect(
            self.decisionTreePredict)
        self.Extratreeclassifier_pushButton.clicked.connect(
            self.extraTreePredict)
        self.bernoulli_pushButton.clicked.connect(self.bernouliPredict)
        self.gaussien_pushButton.clicked.connect(self.gaussienPredict)
        self.multinomial_pushButton.clicked.connect(self.multinomialPredict)
        self.mlp_pushButton.clicked.connect(self.mlpPredict)
        self.afficherButton.clicked.connect(self.setDefault)

    def openFile(self):

        nom_fichier = QFileDialog.getOpenFileName()
        self.path = nom_fichier[0]
        pathx = self.path

        pixmap = QtGui.QPixmap(pathx)
        pixmap4 = pixmap.scaled(151, 301, QtCore.Qt.KeepAspectRatio)
        self.image_test_label.setPixmap(QtGui.QPixmap(pixmap4))

        imageTestRGB = imread(self.path)
        self.imageTest = rgb2gray(imageTestRGB)
        thresh = threshold_otsu(self.imageTest)
        binaryTest = self.imageTest > thresh
        self.imageTest = np.ravel(binaryTest)

    def decisionTreePredict(self):

        y_pred = decisionTree.predict([self.imageTest])
        y_pred = np.int(y_pred[0])
        self.field_test_Decisiontreeclassifier.setText(self.alphabet[y_pred-1])

    def extraTreePredict(self):

        y_pred = extraTree.predict([self.imageTest])
        y_pred = np.int(y_pred[0])
        self.field_test_Extratreeclassifier.setText(self.alphabet[y_pred-1])

    def bernouliPredict(self):

        y_pred = bernoulli.predict([self.imageTest])
        y_pred = np.int(y_pred[0])
        self.field_test_bernoulli.setText(self.alphabet[y_pred-1])

    def gaussienPredict(self):

        y_pred = gaussien.predict([self.imageTest])
        y_pred = np.int(y_pred[0])
        self.field_test_gaussien.setText(self.alphabet[y_pred-1])

    def multinomialPredict(self):

        y_pred = multinomial.predict([self.imageTest])
        y_pred = np.int(y_pred[0])
        self.field_test_Multinomial.setText(self.alphabet[y_pred-1])

    def mlpPredict(self):

        y_pred = mlpcl.predict([self.imageTest])
        y_pred = np.int(y_pred[0])
        self.field_test_mlp.setText(self.alphabet[y_pred-1])

    def setDefault(self):

        self.taux_reconnaissance_DECT.setText(
            str(float("{:.2f}".format(taux_app_decisionTree)))+"%")
        self.taux_reconnaissance_EXTT.setText(
            str(float("{:.2f}".format(taux_app_extraTree)))+"%")
        self.taux_reconnaissance_BERN.setText(
            str(float("{:.2f}".format(taux_app_bernoulli)))+"%")
        self.taux_reconnaissance_GAUSS.setText(
            str(float("{:.2f}".format(taux_app_gaussien)))+"%")
        self.taux_reconnaissance_MULTI.setText(
            str(float("{:.2f}".format(taux_app_multinomial)))+"%")
        self.taux_reconnaissance_MLP.setText(
            str(float("{:.2f}".format(taux_app_mlpcl)))+"%")

        self.temps_execution_DECT.setText(
            str(float("{:.3f}".format(temps_exec_decisionTree))))
        self.temps_execution_EXTT.setText(
            str(float("{:.3f}".format(temps_exec_extraTree))))
        self.temps_execution_BERN.setText(
            str(float("{:.3f}".format(temps_exec_bernoulli))))
        self.temps_execution_GAUSS.setText(
            str(float("{:.3f}".format(temps_exec_gaussien))))
        self.temps_execution_MULTI.setText(
            str(float("{:.3f}".format(temps_exec_multinomial))))
        self.temps_execution_MLP.setText(
            str(float("{:.3f}".format(temps_exec_mlpcl))))

        self.taux_test_DECT.setText(
            str(float("{:.2f}".format(taux_test_decisionTree))) + "%")
        self.taux_test_EXTT.setText(
            str(float("{:.2f}".format(taux_test_extraTree))) + "%")
        self.taux_test_BERN.setText(
            str(float("{:.2f}".format(taux_test_berno))) + "%")
        self.taux_test_GAUS.setText(
            str(float("{:.2f}".format(taux_test_gauss))) + "%")
        self.taux_test_MULTI.setText(
            str(float("{:.2f}".format(taux_test_multi))) + "%")
        self.taux_test_MLP.setText(
            str(float("{:.2f}".format(taux_test_multi))) + "%")


if __name__ == "__main__":
    import sys

    # initialisation

    # images d'apprentissage traitées
    images_apprentissage = pre_traitement("apprentissage/", 62)
    # images de test traitées
    images_test = pre_traitement("test/", 41)

    # etiquettes des images d'apprentissage
    d_apprentissage = d_function(1)

    # etiquettes des images de test
    d_test = d_function(0)

    # apprentissage et temps d'execution

    decisionTree, temps_exec_decisionTree = learning_DecisionTreeClassifier(
        images_apprentissage, d_apprentissage)
    extraTree, temps_exec_extraTree = learning_ExtraTreeClassifier(
        images_apprentissage, d_apprentissage)
    bernoulli, temps_exec_bernoulli = learning_bernoulli(
        images_apprentissage, d_apprentissage)
    gaussien, temps_exec_gaussien = learning_gaussien(
        images_apprentissage, d_apprentissage)
    multinomial, temps_exec_multinomial = learning_multinomial(
        images_apprentissage, d_apprentissage)
    mlpcl, temps_exec_mlpcl = learning_algorithme(
        images_apprentissage, d_apprentissage)

    taux_app_decisionTree = taux(
        images_apprentissage, decisionTree, d_apprentissage)
    taux_app_extraTree = taux(images_apprentissage, extraTree, d_apprentissage)
    taux_app_bernoulli = taux(images_apprentissage, bernoulli, d_apprentissage)
    taux_app_gaussien = taux(images_apprentissage, gaussien, d_apprentissage)
    taux_app_multinomial = taux(
        images_apprentissage, multinomial, d_apprentissage)
    taux_app_mlpcl = taux(
        images_apprentissage, mlpcl, d_apprentissage)

    taux_test_decisionTree = taux(images_test, decisionTree, d_test)
    taux_test_extraTree = taux(images_test, extraTree, d_test)
    taux_test_berno = taux(images_test, bernoulli, d_test)
    taux_test_gauss = taux(images_test, gaussien, d_test)
    taux_test_multi = taux(images_test, multinomial, d_test)
    taux_test_mlpcl = taux(images_test, mlpcl, d_test)

    # lancement

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
