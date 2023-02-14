from flask import Flask, flash, redirect, render_template, request, session, abort
app = Flask(__name__)

@app.route("/")
def index():
    return "Index!"

@app.route("/validate_answer")
def validate_answer():
    send = {
   "validate_answer": [

      {
         "question_text": "Explain why it is easy to have constant temperature for a salt bath furnace",
         "answer_text": "A salt bath furnace is able to maintain a constant temperature because the salt bath acts as a heat storage medium."
      }
   ]
}
    return send

@app.route("/members")
def members():
    return "Members"

@app.route("/members/<string:name>/")
def getMember(name):
    return render_template(
    'test.html',name=name)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)