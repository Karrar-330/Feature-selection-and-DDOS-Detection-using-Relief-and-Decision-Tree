from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from skrebate import ReliefF
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

Ui_MainWindow, _ = uic.loadUiType("1.ui")
Ui_MainWindow2, _ = uic.loadUiType("2.ui")
Ui_MainWindow3, _ = uic.loadUiType("3.ui")
Ui_MainWindow4, _ = uic.loadUiType("4.ui")
Ui_MainWindow5, _ = uic.loadUiType("5.ui")
Ui_MainWindow6, _ = uic.loadUiType("6.ui")
Ui_MainWindow7, _ = uic.loadUiType("7.ui")
path='E:\\project\\test'

class MyGui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.start = self.findChild(QtWidgets.QPushButton, 'start')
        self.start.clicked.connect(self.button_clicked)
        self.show()

    def button_clicked(self):
        self.gui = QtWidgets.QMainWindow()
        self.qq = Ui_MainWindow2()
        self.qq.setupUi(self.gui)
        self.gui.show()
        self.gui.browsebtn = self.gui.findChild(QtWidgets.QPushButton, 'browse')
        self.gui.browsebtn.clicked.connect(self.Browse)
        self.gui.bronext = self.gui.findChild(QtWidgets.QPushButton, 'brownext')
        self.gui.bronext.clicked.connect(self.Bronext)

        self.window().close()

    def Browse(self):
        global fname
        fname = QFileDialog.getOpenFileName(self, 'open file', path, 'xml file (*.csv)')
        df = pd.read_csv(fname[0])

        self.gui.labelfile = self.gui.findChild(QtWidgets.QTextEdit, 'brotext')
        self.gui.fea = self.gui.findChild(QtWidgets.QTextEdit, 'feana')
        self.gui.labelfile.setText ( 'File Name : ' + (fname[0].split('/')[len(fname[0].split('/')) - 1]) + '\n \n' +
    'Instance :' + str(df.shape[0]) + '\n \n' +
    'Features :' + str(df.shape[1]))
        clum=''
        num=0
        for i in df.columns:
            num += 1
            clum += str(num)+' - '+ i + '\n'
        self.gui.fea.setText(str(clum))


    def Bronext(self):
        self.gui = QtWidgets.QMainWindow()
        self.pre = Ui_MainWindow3()
        self.pre.setupUi(self.gui)
        self.gui.show()
        self.gui.Prepro = self.gui.findChild(QtWidgets.QPushButton, 'prebtn')
        self.gui.Prepro.clicked.connect(self.Preprocess)
        self.gui.Prenext = self.gui.findChild(QtWidgets.QPushButton, 'prenext')
        self.gui.Prenext.clicked.connect(self.prenext)

    def Preprocess(self):
        global df
        df = pd.read_csv(fname[0])
        df = df.dropna()
        df.replace('BENIGN', 0, inplace=True)
        df.replace('DDoS', 1, inplace=True)

        df = df.apply(lambda x: round((x - np.min(x)) / (np.max(x) - np.min(x)), 3))
        df = df.dropna(axis=1, how='any')
        clum = ''
        num = 0
        for i in df.columns:
            num += 1
            clum += str(num) + ' - ' + i + '\n'
        self.gui.prete = self.gui.findChild(QtWidgets.QTextEdit, 'precul')
        self.gui.prete.setText(str(clum))
        self.gui.prerow = self.gui.findChild(QtWidgets.QTextEdit, 'prerow')
        self.gui.prerow.setText(str(df.shape[0]))
        self.gui.preclu = self.gui.findChild(QtWidgets.QTextEdit, 'preclu')
        self.gui.preclu.setText( str(df.shape[1]))
        return df

    def prenext(self):
        self.gui = QtWidgets.QMainWindow()
        self.rel = Ui_MainWindow4()
        self.rel.setupUi(self.gui)
        self.gui.show()
        self.gui.relptn = self.gui.findChild(QtWidgets.QPushButton, 'reliefbtn')
        self.gui.relptn.clicked.connect(self.relief)
        self.gui.relnext = self.gui.findChild(QtWidgets.QPushButton, 'reliefnext')
        self.gui.relnext.clicked.connect(self.Relnext)

    def relief(self):
        self.gui.selelab1 = self.gui.findChild(QtWidgets.QTextEdit, 'fea')
        selection = pd.DataFrame()

        attack = (df.loc[(df[' Label'] == 1)]).head(5500)
        normal = df.loc[(df[' Label'] == 0)].head(5500)
        selection = attack
        selection = selection.append(normal)
        X = selection.drop(' Label', axis=1).values
        y = selection[' Label'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67)
        
        relief = ReliefF(n_neighbors=20)
        relief.fit(X_train, y_train)
        feature_weights = relief.feature_importances_

        top_features = feature_weights.argsort()[-5:][::-1]
        names = df.columns[top_features]
        global bstfchr
        bstfchr = pd.DataFrame()
        for i in names:
            bstfchr[i] = df[i]
        bstfchr[' Label'] = df[' Label']

        self.gui.selelab1.setText('The selected Features : ' + '\n'+
                                  str(names[0] + '\n' +
                                      names[1] + '\n' +

                                      names[2] + '\n' +

                                      names[3] + '\n' +
                                      names[4] + '\n') )

        self.gui.rerow = self.gui.findChild(QtWidgets.QTextEdit, 'rerow')
        self.gui.rerow.setText(str(bstfchr.shape[0]))
        self.gui.reclu = self.gui.findChild(QtWidgets.QTextEdit, 'reclu')
        self.gui.reclu.setText(str(bstfchr.shape[1] - 1))
        return bstfchr

    def Relnext(self):
        self.gui = QtWidgets.QMainWindow()
        self.dt = Ui_MainWindow5()
        self.dt.setupUi(self.gui)
        self.gui.show()
        self.gui.warbtn = self.gui.findChild(QtWidgets.QPushButton, 'warbtn')
        self.gui.warbtn.clicked.connect(self.War)
        self.gui.wrnext = self.gui.findChild(QtWidgets.QPushButton, 'wrnext')
        self.gui.wrnext.clicked.connect(self.Warnext)

    def War(self):

        global df
        df = bstfchr.drop(bstfchr.columns[0], axis=1)
        df = df.drop(df.columns[0], axis=1)

        self.gui.deslab = self.gui.findChild(QtWidgets.QLabel, 'deslab')
        self.gui.deslab.setText(
            'Data after Wrapper feature selection' + '\n \n' + 'Instance :' + str(df.shape[0]) + '\n \n' +
            str(df.columns.tolist()[0]+'\n'+df.columns.tolist()[1]+'\n'+df.columns.tolist()[2]))
        df = df[(df != 0).any(axis=1)]
        self.gui.deslab2 = self.gui.findChild(QtWidgets.QLabel, 'deslab2')
        self.gui.deslab2.setText(
            'Data after Cleaning' + '\n \n' +
            'Features :' + str(df.shape[1]-1) + ' \n')

        self.gui.wrow = self.gui.findChild(QtWidgets.QTextEdit, 'wrow')
        self.gui.wrow.setText(str(df.shape[0]))
        self.gui.wclu = self.gui.findChild(QtWidgets.QTextEdit, 'wclu')
        self.gui.wclu.setText(str(df.shape[1]-1))
        return df


    def Warnext(self):
        self.gui = QtWidgets.QMainWindow()
        self.dt = Ui_MainWindow6()
        self.dt.setupUi(self.gui)
        self.gui.show()
        self.gui.desbtn = self.gui.findChild(QtWidgets.QPushButton, 'desbtn')
        self.gui.desbtn.clicked.connect(self.Det)
        self.gui.end = self.gui.findChild(QtWidgets.QPushButton, 'd3next')
        self.gui.end.clicked.connect(self.D3next)
    def Det(self):

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        global vv
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67)


        clf = DecisionTreeClassifier()

        vv=clf.fit(X_train, y_train)

        pred = vv.predict(X_test)
        p=classification_report(y_test, pred)
        cm = confusion_matrix(y_test, pred)

        score = accuracy_score(y_test, pred)
        self.gui.deslab3 = self.gui.findChild(QtWidgets.QTextEdit, 'desl')
        self.gui.deslab3.setText('Attack detection' + '\n \n \n' +
                                 'Model Evaluation : hold out' + '\t \t' + 'Train_size : 0.67' + '\n \n' +
                                 'Accuracy :' + str(round(score, 3)) + '\n \n\n' +str(p)+ '\n \n\n' +"Confusion Matrix:\n"+str(cm))
    def D3next(self):
        self.gui = QtWidgets.QMainWindow()
        self.dt = Ui_MainWindow7()
        self.dt.setupUi(self.gui)
        self.gui.show()
        self.gui.desbtn = self.gui.findChild(QtWidgets.QPushButton, 'decision')
        self.gui.desbtn.clicked.connect(self.only_one)
        self.gui.browf = self.gui.findChild(QtWidgets.QPushButton, 'browf')
        self.gui.browf.clicked.connect(self.Browf)
        self.gui.end = self.gui.findChild(QtWidgets.QPushButton, 'end')
        self.gui.end.clicked.connect(self.End)
    def Browf(self):
        global df1
        fname = QFileDialog.getOpenFileName(self, 'open file', path, 'xml file (*.csv)')
        df1 = pd.read_csv(fname[0])


    def only_one(self):

        self.gui.label = self.gui.findChild(QtWidgets.QLabel, 'label')

        pred = vv.predict(df1.values)

        if pred[0] ==1:
            self.gui.label.setText('Attack')
        else:
            self.gui.label.setText('Normal')


    def End(self):
        self.gui.window().close()


app = QApplication([])
gui = MyGui()

app.exec_()
