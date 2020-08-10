#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Personal Assitant Enabled AI

import speech_recognition as sr  
import playsound    
from gtts import gTTS   
import os  
import wolframalpha 
from selenium import webdriver  
from selenium.webdriver.common.keys import Keys
from io import BytesIO
from io import StringIO
import smtplib

num = 1


def talk(audioString):
    global num
    num +=1
    print("You : ", audioString)
    toSpeak = gTTS(text=audioString, lang='en-US', slow=False)
    file = str(num)+".mp3"
    toSpeak.save(file)
    '''mixer.init()
    mixer.music.load('D:\Speech\\audio\spoken.mp3')
    mixer.music.play()
    time.sleep(5)
    mixer.music.stop()'''
    # song = AudioSegment.from_file(mp3_fp, format="mp3")
    # playsound.playsound(mp3_fp)
    playsound.playsound(file, True)
    os.remove(file)


def readCommand():
    r = sr.Recognizer()
    audio = ''
    with sr.Microphone() as source:
        print("Speak...")
        audio = r.listen(source, phrase_time_limit=5)
    print("Stop.")
    try:
        text = r.recognize_google(audio,language='en-US')
        print("You : ", text)
        return text
    except:
        talk("Could not understand your audio, PLease try again!")
        return 0


def internetSearch(input):
    driver = webdriver.Firefox()
    driver.implicitly_wait(1)
    driver.maximize_window()
    if 'youtube' in input.lower():
        talk("Opening in youtube")
        indx = input.lower().split().index('youtube')
        query = input.split()[indx+1:]
        driver.get("http://www.youtube.com/results?search_query=" + '+'.join(query))
        return

    elif 'wikipedia' in input.lower():
        talk("Opening Wikipedia")
        indx = input.lower().split().index('wikipedia')
        query = input.split()[indx + 1:]
        driver.get("https://en.wikipedia.org/wiki/" + '_'.join(query))
        return
    else:
        if 'google' in input:
            indx = input.lower().split().index('google')
            query = input.split()[indx + 1:]
            driver.get("https://www.google.com/search?q=" + '+'.join(query))
        elif 'search' in input:
            indx = input.lower().split().index('google')
            query = input.split()[indx + 1:]
            driver.get("https://www.google.com/search?q=" + '+'.join(query))
        else:
            driver.get("https://www.google.com/search?q=" + '+'.join(input.split()))
        return


def process_input(input):
    try:
        if "tell me about yourself" in input or "brief me about you" in input:
            speak = '''Hello, I am Person. Your personal Assistant.
            I am here to make your life easier. 
            i can perform various tasks such as sum calculations or opening Apps etcetra as per your command'''
            talk(speak)
            return
        elif "who is your creator" in input or "who created" in input:
            speak = "I have been created by Narinder."
            talk(speak)
            return
        elif "From where you belong" in input:
            speak = """I belong to india."""
            talk(speak)
            return
        elif "calculate" in input.lower():
            app_id= "E46YXW-T5LG6RT7K7"
            client = wolframalpha.Client(app_id)

            indx = input.lower().split().index('calculate')
            query = input.split()[indx + 1:]
            res = client.query(' '.join(query))
            answer = next(res.results).text
            talk("The answer is " + answer)
            return
        elif 'open' in input:
            openApp(input.lower())
            return
        elif 'search' in input or 'play' in input:
            internetSearch(input.lower())
            return
        elif 'email' in input or 'send message' in input:
            s = smtplib.SMTP('smtp.gmail.com', 587) 
            s.starttls() 
            s.login("sender_email_id", "sender_email_id_password") 
            message = "Message_you_need_to_send"
            s.sendmail("sender_email_id", "receiver_email_id", message) 
            s.quit() 
        else:
            talk("I can perform web search for you, Can i do it now?")
            ans = readCommand()
            if 'yes' in str(ans) or 'yeah' in str(ans):
                internetSearch(input)
            else:
                return
    except Exception as e:
        print(e)
        talk("I don't understand, I can perform web search for you, Can i do it now?")
        ans = readCommand()
        if 'yes' in str(ans) or 'yeah' in str(ans):
            internetSearch(input)


if __name__ == "__main__":
    #talk("What's your name, Human?")
    name ='Guest'
    #name = readCommand()
    talk("Hello, " + name + '.')
    while(1):
        talk("How can i help you?")
        text = readCommand().lower()
        if text == 0:
            continue
        #talk(text)
        if "exit" in str(text) or "bye" in str(text) or "go " in str(text) or "sleep" in str(text):
            talk("Ok bye, "+ name+'.')
            break
        process_input(text)


# In[ ]:




