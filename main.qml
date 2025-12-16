import QtQuick
import QtQuick.Controls

Rectangle {
    id: frame
    width: 1280
    height: 720
    color: "#6a994e"
    radius: 0

    Rectangle {
        id: nav
        x: 0
        width: 1280
        height: 89
        color: "#386641"
        border.color: "#000000"
        border.width: 0
        anchors.top: parent.top
        anchors.topMargin: 0

        Text {
            id: title
            x: 16
            y: 23
            width: 895
            height: 42
            color: "#f2e8cf"
            text: qsTr("Santa’s Intelligent Gift Recommendation System")
            font.pixelSize: 32
            font.family: "Verdana"
            font.styleName: "ExtraCondensed ExtraBold Italic"
            font.bold: true
        }

        Rectangle {
            id: tinselContainer
            x: 0
            y: 31
            width: 1280
            height: 124
            color: "transparent"

            Repeater {
                model: Math.ceil(tinselContainer.width / 200) // Adjust 200 to the width of your image
                delegate: Image {
                    source: "images/AdobeStock_1023592969.png"
                    width: 1280 // Set this to the actual width of your image
                    height: tinselContainer.height
                    smooth: false
                }
            }
        }

        Text {
            id: versionnumber
            x: 1156
            y: 31
            width: 109
            height: 27
            color: "#c7c2b8"
            text: qsTr("V 1.0.0")
            font.pixelSize: 25
            font.styleName: "ExtraCondensed ExtraBold Italic"
            font.family: "Verdana"
            font.bold: true
        }
    }

    Rectangle {
        id: nameDiv
        y: 176
        width: 564
        height: 76
        color: "#bc4749"
        radius: 10
        border.color: "#a7c957"
        border.width: 5
        anchors.left: parent.left
        anchors.leftMargin: 34

        TextInput {
            id: nameInput
            x: 241
            y: 18
            width: 285
            height: 45
            color: "#f2e8cf"
            font.pixelSize: 28
            horizontalAlignment: Text.AlignLeft
            verticalAlignment: Text.AlignVCenter
            padding: 3
            focus: true // Set initial focus here
        }

        Rectangle {
            id: nameUnderline
            x: 230
            y: 57
            width: 315
            height: 6
            color: "#f2e8cf"
            radius: 5
            border.color: "#f2e8cf"
            border.width: 3
        }

        Text {
            id: nameLabel
            x: 21
            y: 23
            width: 204
            height: 30
            color: "#f2e8cf"
            text: qsTr("Child's Name")
            font.pixelSize: 28
            font.styleName: "ExtraCondensed ExtraBold Italic"
            font.family: "Verdana"
            font.bold: true
        }
    }

    Rectangle {
        id: output
        y: 176
        width: 500
        height: 480
        color: "#bc4749"
        radius: 10
        border.color: "#a7c957"
        border.width: 5
        anchors.right: parent.right
        anchors.rightMargin: 71

        Text {
            id: outputLabel
            x: 172
            y: 16
            width: 156
            height: 53
            color: "#f2e8cf"
            text: qsTr("Output")
            font.pixelSize: 38
            font.styleName: "ExtraCondensed ExtraBold Italic"
            font.family: "Verdana"
            font.bold: true
        }

        ScrollView {
            id: gifts
            width: 439
            height: 369
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.rightMargin: 30
            anchors.topMargin: 81
            font.family: "Verdana"

            Column {
                id: column
                x: 0
                y: 0
                width: 439
                height: 369

                Label {
                    id: label
                    color: "#f2e8cf"
                    text: qsTr("•gift")
                    font.family: "Verdana"
                    font.pointSize: 35
                }
            }
        }
    }

    Rectangle {
        id: ageDiv
        y: 282
        width: 564
        height: 76
        color: "#bc4749"
        radius: 10
        border.color: "#a7c957"
        border.width: 5
        anchors.left: parent.left
        anchors.leftMargin: 34

        TextInput {
            id: ageInput
            x: 241
            y: 18
            width: 285
            height: 45
            color: "#f2e8cf"
            font.pixelSize: 28
            horizontalAlignment: Text.AlignLeft
            verticalAlignment: Text.AlignVCenter
            padding: 3
        }

        Rectangle {
            id: ageUnderline
            x: 230
            y: 57
            width: 315
            height: 6
            color: "#f2e8cf"
            radius: 5
            border.color: "#f2e8cf"
            border.width: 3
        }

        Text {
            id: ageLabel
            x: 17
            y: 23
            width: 204
            height: 30
            color: "#f2e8cf"
            text: qsTr("Child's Age")
            font.pixelSize: 28
            font.styleName: "ExtraCondensed ExtraBold Italic"
            font.family: "Verdana"
            font.bold: true
        }
    }

    Rectangle {
        id: idDiv
        y: 383
        width: 564
        height: 76
        color: "#bc4749"
        radius: 10
        border.color: "#a7c957"
        border.width: 5
        anchors.left: parent.left
        anchors.leftMargin: 34

        TextInput {
            id: idInput
            x: 241
            y: 18
            width: 285
            height: 45
            color: "#f2e8cf"
            font.pixelSize: 28
            horizontalAlignment: Text.AlignLeft
            verticalAlignment: Text.AlignVCenter
            padding: 3
        }

        Rectangle {
            id: idUnderline
            x: 230
            y: 57
            width: 315
            height: 6
            color: "#f2e8cf"
            radius: 5
            border.color: "#f2e8cf"
            border.width: 3
        }

        Text {
            id: idLabel
            x: 17
            y: 23
            width: 204
            height: 30
            color: "#f2e8cf"
            text: qsTr("Child's ID")
            font.pixelSize: 28
            font.styleName: "ExtraCondensed ExtraBold Italic"
            font.family: "Verdana"
            font.bold: true
        }
    }

    Button {
        id: genBtn
        y: 492
        width: 564
        height: 164
        text: qsTr("Generate")
        anchors.left: parent.left
        anchors.leftMargin: 34
        font {
            family: "Verdana"
            pixelSize: 48
        }
        background: Rectangle {
            id: buttonBackground
            color: genBtn.pressed ? "#a63d40" : (genBtn.hovered ? "#bc4749" : "#bc4749")
            radius: 15
            border.color: "#a7c957"
            border.width: 5
        }

        MouseArea {
            id: buttonMouseArea
            anchors.fill: parent
            hoverEnabled: true
            onClicked: {
                console.log("Button clicked!");
            }
            onPressed: {
                buttonBackground.color = "#a63d40"; // Change color on press
            }
            onReleased: {
                buttonBackground.color = "#d1603d"; // Change color back on release
            }
            onEntered: {
                buttonBackground.color = "#d1603d"; // Change color on hover
            }
            onExited: {
                buttonBackground.color = "#bc4749"; // Reset color when hover ends
            }
        }
    }

    // Define Tab navigation order
    Keys.onPressed: {
        if (event.key === Qt.Key_Tab) {
            if (nameInput.focus) {
                ageInput.forceActiveFocus();
            } else if (ageInput.focus) {
                idInput.forceActiveFocus();
            } else if (idInput.focus) {
                genBtn.forceActiveFocus();
            } else {
                nameInput.forceActiveFocus();
            }
            event.accepted = true;
        }
    }
}
