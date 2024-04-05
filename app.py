from flask import Flask,request,render_template
import pickle

app=Flask(__name__)
@app.route("/")
def Homepage():
    return render_template("frontend_page.html")

@app.route("/get_test_data")
def get_data():
    
    load_model=pickle.load(open(r"logistic_model.pkl","rb"))
    
    target=['setosa', 'versicolor', 'virginica']
    
    sepal_length = 4.9
    sepal_width = 3.0
    petal_length = 1.4
    petal_width = 0.2
    
    result=load_model.predict([[sepal_length,sepal_width,petal_length,petal_width]])

    return f"your prediction will be : {target[result[0]]}"

@app.route("/get_user_data",methods=["POST"])
def get_user_data():

    data=request.form
    print(f"data : {data}")

    load_model=pickle.load(open(r"logistic_model.pkl","rb"))

    target=['setosa', 'versicolor', 'virginica']

    sepal_length = eval(data["sepal_length"])
    sepal_width = eval(data["sepal_width"])
    petal_length = eval(data["petal_length"])
    petal_width = eval(data["petal_width"])

    result=load_model.predict([[sepal_length,sepal_width,petal_length,petal_width]])

    return f"your desired prediction is : {target[result[0]]}"

if __name__=="__main__": 
    app.run(debug=True)
