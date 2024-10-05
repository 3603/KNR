from streamlit_mic_recorder import mic_recorder # pour enregistrer l'audio
import streamlit as st # pour le rendering sur page web
import io # pour pouvoir manipuler le flux audio
from speechbrain.pretrained import EncoderASR # indispensable au modèle speechbrain
import os # pour pouvoir ouvrir/fermer un fichier
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def BrainSTT(start_prompt="Enregistrer",stop_prompt="Arrêter",just_once=False,use_container_width=False,language=None,callback=None,args=(),kwargs={}):
    if not '_last_speech_to_text_transcript_id' in st.session_state:
        st.session_state._last_speech_to_text_transcript_id=0
    if not '_last_speech_to_text_transcript' in st.session_state:
        st.session_state._last_speech_to_text_transcript=None
    audio = mic_recorder(start_prompt=start_prompt,stop_prompt=stop_prompt,just_once=just_once,use_container_width=use_container_width)
    new_output=False
    if audio is None:
        output=None
    else:
        st.write("audio n'est pas none")
        st.write("id audio : " + audio[id])
        id=audio['id']
        st.write("id = audio[id]")
        new_output=(id>st.session_state._last_speech_to_text_transcript_id)
        st.write("new_output = id>st.session_state._last_speech_to_text_transcript_id")
        if new_output:
            st.write("new_output n'est pas null")
            output=None
            st.write("output = None")
            st.session_state._last_speech_to_text_transcript_id=id
            st.write("st.session_state._last_speech_to_text_transcript_id = id")
            audio_BIO = io.BytesIO(audio['bytes'])
            st.write("audio_BIO se remplit")
            audio_BIO.name='audio.mp3'
            st.write("audio_BIO.name = 'audio.mp3'")
            success=False
            st.write("success = False")
            err=0
            st.write("err =" + err)
            while not success and err<3: #Retry up to 3 times in case ...
                try:
                    st.write("début du try")
                    transcript = st.session_state.openai_client.audio.transcriptions.create(
                        model="speechbrain/asr-wav2vec2-commonvoice-fr",
                        file=audio_BIO,
                        language=language
                    )
                    st.write("transcript en cours")
                except Exception as e:
                    print(str(e)) # log the exception in the terminal
                    err+=1
                else:
                    st.write("transcription finie")
                    success=True
                    st.write ("success = " + success)
                    output=transcript.text
                    st.write("output = transcript.text")
                    st.session_state._last_speech_to_text_transcript=output
                    st.write("st.session_state._last_speech_to_text_transcript = output")
        elif not just_once:
            output=st.session_state._last_speech_to_text_transcript
            st.write("output = st.session_state._last_speech_to_text_transcript")
        else:
            output=None
            st.write("dernier output = None")
    if new_output and callback:
        st.write("new_output and callback:")
        callback(*args,**kwargs)
    return output
    

text=BrainSTT(language='fr')
if text:
    st.write(text)