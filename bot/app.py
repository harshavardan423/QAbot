
from flask import Flask, request, jsonify,render_template

app = Flask(__name__)
from bot2 import ask,score
from bot3 import summarize
import bot3

@app.route('/')
def index():
    return render_template('index.html')
##ADMIN ROUTES                  
@app.route('/validate_answer', methods=['GET','POST'])
def validate_answer():  
    if request.method == 'POST':
        message = request.form['message']
        
        import dm_2
        output = str( ask(f"{message}"))
       
        if output != "fail" :  #dm_2.generate_words_list(output)
            response = score(message,output)
            return ( "Score : " + str(response))
        else :
            return ("Failed")
    else:
        return render_template('validate_answer.html')
    
@app.route('/validate_answer2', methods=['GET','POST'])
def validate_answer2():  
    if request.method == 'POST':
        message = request.form['message']
        perfect_answer = request.form['perfect_answer']
        
        import dm_2
        # output = str( ask(f"{message}"))
        # if output != "fail" :  #dm_2.generate_words_list(output)
        
        response = score(message,perfect_answer)    
        return  jsonify( matchingkeywords = str(response[0]), matching_sentences = response[1], missing_words=response[4], missing_sentences = response[3],grammar = response[2],prompt = response[7] )
        # else :
        #     return ("Failed")
    else:
        return render_template('validate_answer2.html')

@app.route('/new_test', methods=['POST'])
def new_test():
    return 'upload_new_test'

@app.route('/tests_dashboard', methods=['POST'])
def tests_dashboard():
    return 'tests_dashboard'

## USER ROUTES

@app.route('/user_tests_dashboard', methods=['POST'])
def user_tests_dashboard():
    return 'user_tests_dashboard'


@app.route('/learn_with_askbot', methods=['GET','POST'])
def learn_with_askbot():
    if request.method == 'POST':
        message = request.form['message']
    
        response = ask(f"Summarize {message}")
        return response
    else:
        return render_template('chatbox.html')
    

@app.route('/summarize_topic', methods=['GET','POST'])
def summarize_topic():
    if request.method == 'POST':
        message = request.form['message']
    
        response = bot3.ask(f"{message}")
        print(response)
        return response
    else:
        return render_template('summarize.html')

## BASIC AUTH ROUTES

@app.route('/login')
def login():
    return 'login'

@app.route('/logout')
def logout():
    return 'logout'

@app.route('/register')
def register():
    return 'register'




if __name__ == '__main__':
    app.run(port=8000, debug=False, host='0.0.0.0')
   