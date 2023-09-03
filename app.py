from urllib import request
from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo.mongo_client import MongoClient
import random
import shortuuid
import json
from datetime import datetime
import cv2
import numpy as np
import os
from pdf2image import convert_from_path
import fitz
from paddleocr import PaddleOCR,draw_ocr
from spellchecker import SpellChecker
from sentence_transformers import SentenceTransformer, util

uri = "mongodb+srv://admin:admin@cluster0.wonbr.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri)


app = Flask(__name__)

cors = CORS(app)




@app.route('/api/register', methods=["POST"])
def register():
    global client
    db = client["hackclassroom"]
    col = db["users"]
    
    if request.method == "POST":
        req = request.json
        x = col.find_one({"username": req["username"]})
        if x:
            return ({"status":"⚠️ Already Exists"})
        else:
            col.insert_one(req)
            return ({"status":"✅ Success"})
    return jsonify({})

@app.route('/api/login', methods=["POST"])
def login():
    global client
    db = client["hackclassroom"]
    col = db["users"]
    
    if request.method == "POST":
        req = request.json
        x = col.find_one({"username": req["username"], "password":req["password"]})
        print(x)
        if x:
            return ({"status":"✅ Success", "role":x["isTeacher"]})
        else:
            return ({"status":"⚠️ Incorrect Password or Not Registered"})
    return jsonify({})

@app.route('/api/addclass', methods=["POST"])
def addclass():
    global client
    db = client["hackclassroom"]
    col = db["classes"]
    
    class_images = ["https://i.ibb.co/K7rydcW/image.png", "https://i.ibb.co/sJckdfj/image.png", "https://i.ibb.co/sC6N7mD/image.png", "https://i.ibb.co/8Dp3xRS/image.png", "https://i.ibb.co/Hg34jbT/image.png", "https://i.ibb.co/tC4RySy/image.png"]

    if request.method == "POST":
        req = request.json
        print(req)
        req["img"] = random.choice(class_images)
        col.insert_one(req)
    return jsonify({})

@app.route('/api/getclass', methods=["GET"])
def getclass():
    global client
    db = client["hackclassroom"]
    col = db["classes"]

    if request.method == "GET":
        data = []
        x = col.find()
        for ele in x:
            print(ele)
            del ele["_id"]
            data.append(ele)

    return {"data":data}

@app.route('/api/getquestions', methods=["POST"])
def getquestions():
    global client
    db = client["hackclassroom"]
    col = db["questions"]

    if request.method == "POST":
        req = request.json
        data = []
        x = col.find()
        for ele in x:
            if (ele["subject"] == req["subject"]):
                del ele["_id"]
                data.append(ele)
    return {"data":data}

@app.route('/api/addquestions', methods=["POST"])
def addquestions():
    global client
    db = client["hackclassroom"]
    col = db["questions"]

    if request.method == "POST":
        req = request.json
        data = []
        print(req)
        x = col.find_one(req)
        print(x)
        if x == None:
            col.insert_one(req)
            return {"status":"✅ Added Successfully"}
        else:
            return {"status":"⚠️ Already added"}

    return {"data":data}

@app.route('/api/getAssignments', methods=["POST"])
def getAssignments():
    global client
    db = client["hackclassroom"]
    col = db["assignments"]

    if request.method == "POST":
        req = request.json
        data = []
        print(req)
        x = col.find()
        print(x)
        for ele in x:
            print(ele)
            if ele["subject"]==req["subject"]:
                del ele["_id"]
                data.append(ele)
        data.sort(key = lambda x:x["date"], reverse = True)
        print(data)
    return {"data":data}

@app.route('/api/addAssignments', methods=["POST"])
def addAssignments():
    global client
    db = client["hackclassroom"]
    col = db["assignments"]

    if request.method == "POST":
        req = request.json
        data = []
        python_date = datetime.now()
        json_date = json.dumps(python_date.isoformat())
        x = col.insert_one({"assignmentID":shortuuid.uuid(),"subject":req["subject"], "assignments":[], "date":python_date})

    return {"status":"Created Successfully 🚀"}

@app.route('/api/makeQuestions', methods=["POST"])
def makeQuestions():
    global client
    db = client["hackclassroom"]
    col = db["assignments"]
    collq = db["questions"]

    if request.method == "POST":
        req = request.json

        x = col.find_one({"assignmentID": req["assignment"]})
        if any(req["username"] in d.values() for d in x["assignments"]):
            
            print("YES")
            for temp in x["assignments"]:
                print(x["assignments"])
                if temp["username"] == req["username"]:
                    if temp["status"]:
                        data = temp["questions"]
                        return {"status":"Attended","data":data}
                    else:
                        data = temp["questions"]
                        return {"status":"Not Attended","data":data}
        else:
            print("NO")
            questionList = {}
            x = col.find()
            for ele in x:
                for q in ele["assignments"]:
                    if q["username"] == req["username"]:
                        #print(q["questions"])
                        for ques in q["questions"]:
                            if ques["question"] in questionList:
                                questionList[ques["question"]] = (questionList[ques["question"]] + int(ques["marksObtained"]))/2
                            else:
                                questionList[ques["question"]] = ques["marksObtained"]
            
            for ele in collq.find():
                if ele["question"] not in questionList:
                    questionList[ele["question"]] = 0
            
            questionList = dict(sorted(questionList.items(), key=lambda item: item[1]))
            
            data = []
            count = 0
            for key, value in questionList.items():
                if count < 4:
                    data.append({"question":key, "marksObtained":value})
                    count += 1
                else:
                    break
            print(data)
            #data = dict(sorted(data.items(), key=lambda item: item[0]))

            query = { "assignmentID": req["assignment"] }
            x = col.find_one(query)
            assignmentList = x["assignments"]
            '''
            {'username': 'Tushar', 'status': True, 'questions': [{'question': 'Define Inertia', 'marksObtained': 3}, {'question': 'Define Friction', 'marksObtained': 2}, {'question': 'What is the difference between speed and velocity?', 'marksObtained': 4.8}]}
            '''
            print(assignmentList)
            assignmentList.append({
                "username": req["username"],
                "status":False,
                "questions":data
            })
            newvalues = { "$set": { "assignments": assignmentList} }

            col.update_one(query, newvalues)

            return {"data":data, "status": "Now created"}

    return {"status":"BLANK"}


@app.route('/api/upload', methods=['POST'])
def upload_file():
    global client
    db = client["hackclassroom"]
    col = db["assignments"]
    collq = db["questions"]

    model = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')

    def integerize(index):
        for i in range(len(index)):
            index[i] = int(index[i])
        return index
        
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']

    if file.filename == '':
        return '⚠️ No selected file', 400

    if file and file.filename.endswith('.pdf'):
        uploaded_file = request.files['json_data']
        spell = SpellChecker()

        json_data = uploaded_file.read()
        parsed_data = json.loads(json_data.decode('utf-8'))
        print(parsed_data)
        file.save(file.filename)

        assignment_id = parsed_data["assignment"]
        stud_username = parsed_data["username"]

        x = col.find_one({"assignmentID":assignment_id})
        temp = x["assignments"]
        print(temp)

        questionsList = []
        for ele in temp:
            if ele["username"] == stud_username:
                questionsList = ele["questions"]

        file_path = file.filename
        doc = fitz.open(file_path)  
        for i, page in enumerate(doc):
            pixmap = page.get_pixmap()  
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.h, pixmap.w, len(pixmap.samples) // (pixmap.h * pixmap.w))
            
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur_image = cv2.GaussianBlur(gray_image, (5, 5), 10)
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
            parameters =  cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)

            markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(blur_image)

            if markerIds is not None:
                cv2.aruco.drawDetectedMarkers(image, markerCorners, markerIds)

            index0 = integerize([markerCorners[np.where(markerIds == 0)[0][0]][0][0][0],markerCorners[np.where(markerIds == 0)[0][0]][0][0][1]])
            index1 = integerize([markerCorners[np.where(markerIds == 1)[0][0]][0][0][0],markerCorners[np.where(markerIds == 1)[0][0]][0][0][1]])
            index2 = integerize([markerCorners[np.where(markerIds == 2)[0][0]][0][0][0],markerCorners[np.where(markerIds == 2)[0][0]][0][0][1]])
            index3 = integerize([markerCorners[np.where(markerIds == 3)[0][0]][0][0][0],markerCorners[np.where(markerIds == 3)[0][0]][0][0][1]])

            image = cv2.line(image, index0, index1, (0, 255, 0), 9)
            image = cv2.line(image, index1, index2, (0, 255, 0), 9)
            image = cv2.line(image, index2, index3, (0, 255, 0), 9)
            image = cv2.line(image, index3, index0, (0, 255, 0), 9)

            original_points = np.array([index0, index1, index2, index3], dtype=np.float32)
            output_width = 2480
            output_height = 3508

            output_points = np.array([(0, 0), (output_width - 1, 0), (output_width - 1, output_height - 1), (0, output_height - 1)], dtype=np.float32)
            M = cv2.getPerspectiveTransform(original_points, output_points)
            warped_image = cv2.warpPerspective(image, M, (output_width, output_height))
            #resized_image = cv2.resize(warped_image, (480, 640))
            #cv2.imshow(f"frame{i}",resized_image)
            #cv2.waitKey(0)

            
            ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
            result = ocr.ocr(warped_image, cls=True)
            recognisedtext = ""
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    recognisedtext = recognisedtext +" "+ line[1][0]
            
            x = collq.find_one({"question": questionsList[i]["question"]})
            sentences = [recognisedtext, x["answer"]]
            embeddings = model.encode(sentences)
            cosine_similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
            #print(recognisedtext, x["answer"])
            print(cosine_similarity)
            numeric_value = cosine_similarity.item()
            if numeric_value <= 0:
                print(0)
                questionsList[i]["marksObtained"] = 0 
            else:
                print(5*numeric_value)
                questionsList[i]["marksObtained"] = 5*numeric_value
            
            print("__________________")

        x = col.find_one({"assignmentID":assignment_id})
        temp = x["assignments"]
        for ele in temp:
            if ele["username"] == stud_username:
                ele["questions"] = questionsList
                ele["status"] = True
        
        myquery = {"assignmentID":assignment_id}
        newvalues = { "$set": { "assignments": temp } }

        col.update_one(myquery, newvalues)
        print(temp)
            
        return f'✅ File uploaded as {file.filename}', 200

    return '⚠️ Invalid file format', 400


if __name__ == '__main__':
    app.run(debug=True)


'''
{"_id":{"$oid":"64f38eac2161ef9993386912"},
"assignmentID":"PfX9GEPrAhZkhRcQZnLZwa",
"subject":"Physics 1",
"assignments":[
  {
    "username":"Tushar",
    "questions":[
        {"question":"Define Inertia","marksObtained":3},
        {"question":"Define Friction","marksObtained":2},
        {"question":"What is the difference between speed and velocity?","marksObtained":4.8}
      ]
  }
  ]}
'''