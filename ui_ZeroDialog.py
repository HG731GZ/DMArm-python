# 本文件由 ZeroDialog.ui 自动生成；修改界面后请运行 generate_ui.py。

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ZeroDialog(object):
    def setupUi(self, ZeroDialog):
        ZeroDialog.setObjectName("ZeroDialog")
        self.layout = QtWidgets.QVBoxLayout(ZeroDialog)
        self.layout.setObjectName("layout")
        self.instructions = QtWidgets.QLabel(ZeroDialog)
        self.instructions.setObjectName("instructions")
        self.layout.addWidget(self.instructions)
        self.jointBox = QtWidgets.QComboBox(ZeroDialog)
        self.jointBox.setObjectName("jointBox")
        self.jointBox.addItem("")
        self.jointBox.addItem("")
        self.jointBox.addItem("")
        self.jointBox.addItem("")
        self.jointBox.addItem("")
        self.jointBox.addItem("")
        self.layout.addWidget(self.jointBox)
        self.buttons = QtWidgets.QHBoxLayout()
        self.buttons.setObjectName("buttons")
        self.cancelButton = QtWidgets.QPushButton(ZeroDialog)
        self.cancelButton.setObjectName("cancelButton")
        self.buttons.addWidget(self.cancelButton)
        self.applyButton = QtWidgets.QPushButton(ZeroDialog)
        self.applyButton.setEnabled(False)
        self.applyButton.setObjectName("applyButton")
        self.buttons.addWidget(self.applyButton)
        self.layout.addLayout(self.buttons)

        self.retranslateUi(ZeroDialog)
        QtCore.QMetaObject.connectSlotsByName(ZeroDialog)

    def retranslateUi(self, ZeroDialog):
        _translate = QtCore.QCoreApplication.translate
        ZeroDialog.setWindowTitle(_translate("ZeroDialog", "零位设定"))
        self.instructions.setText(_translate("ZeroDialog", "将电机停在机械参考位，再选择要设零的关节。"))
        self.jointBox.setPlaceholderText(_translate("ZeroDialog", "选择电机"))
        self.jointBox.setItemText(0, _translate("ZeroDialog", "J1"))
        self.jointBox.setItemText(1, _translate("ZeroDialog", "J2"))
        self.jointBox.setItemText(2, _translate("ZeroDialog", "J3"))
        self.jointBox.setItemText(3, _translate("ZeroDialog", "J4"))
        self.jointBox.setItemText(4, _translate("ZeroDialog", "J5"))
        self.jointBox.setItemText(5, _translate("ZeroDialog", "J6"))
        self.cancelButton.setText(_translate("ZeroDialog", "取消"))
        self.applyButton.setText(_translate("ZeroDialog", "确认设零"))
