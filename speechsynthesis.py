#from gtts import gTTS
import librosa
import os, boto3

defaultRegion = 'us-east-1'
defaultUrl = 'https://polly.us-east-1.amazonaws.com'

def connectToPolly(regionName=defaultRegion, endpointUrl=defaultUrl):
    return boto3.client('polly', region_name=regionName, endpoint_url=endpointUrl)

def speak(polly, text, format='mp3', voice='Brian', filename='word.mp3', speed='medium'):
    formatted_string = '<speak><prosody rate=\"' + speed + '\"><amazon:effect name=\"drc\">' + text + '</amazon:effect></prosody></speak>'
    resp = polly.synthesize_speech(OutputFormat=format, Text=formatted_string, TextType="ssml", VoiceId=voice)
    soundfile = open('word.mp3', 'wb')
    soundBytes = resp['AudioStream'].read()
    soundfile.write(soundBytes)
    soundfile.close()

def get_word_audio(word, speed='medium') :
    polly = connectToPolly()
    speak(polly, word, voice='Joanna', speed=speed)
    (x, sr) = librosa.load('word.mp3')
    return x