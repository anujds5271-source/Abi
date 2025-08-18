# Cell 10.5: Simple Audio Transcription Function (Add before Cell 11)
import azure.cognitiveservices.speech as speechsdk
import requests
import os

def get_access_token():
    """Get Azure access token"""
    try:
        token_url = f"https://login.microsoftonline.com/{databricks_app_tenantid_from_metadata_table}/oauth2/v2.0/token"
        
        payload = {
            'grant_type': 'client_credentials',
            'client_id': databricks_app_clientid_from_metadata_table,
            'client_secret': spn_client_secret_from_dbx_secret,
            'scope': 'https://cognitiveservices.azure.com/.default'
        }
        
        response = requests.post(token_url, data=payload)
        
        if response.status_code == 200:
            return response.json()['access_token']
        else:
            print(f"Token error: {response.text}")
            return None
            
    except Exception as e:
        print(f"Auth error: {str(e)}")
        return None

def transcribe_mp3_simple(mp3_file_path):
    """Simple MP3 transcription using Azure Speech Services"""
    try:
        print(f"Processing MP3: {mp3_file_path}")
        
        # Get access token
        access_token = get_access_token()
        if not access_token:
            print("Failed to get access token")
            return None
        
        # Create speech config
        speech_config = speechsdk.SpeechConfig(endpoint=rag_azure_cognitive_services_end_point_from_metadata_table)
        speech_config.authorization_token = access_token
        speech_config.speech_recognition_language = "en-US"
        
        # Create audio config directly from file
        audio_config = speechsdk.audio.AudioConfig(filename=mp3_file_path)
        
        # Create recognizer
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )
        
        # Setup continuous recognition
        transcribed_parts = []
        recognition_done = False
        
        def handle_recognized(evt):
            if evt.result.text.strip():
                print(f"Recognized: {evt.result.text[:50]}...")
                transcribed_parts.append(evt.result.text)
        
        def handle_session_stopped(evt):
            nonlocal recognition_done
            print("Recognition completed")
            recognition_done = True
        
        def handle_canceled(evt):
            nonlocal recognition_done
            details = evt.result.cancellation_details
            print(f"Recognition canceled: {details.reason}")
            if details.error_details:
                print(f"Error: {details.error_details}")
            recognition_done = True
        
        # Connect events
        recognizer.recognized.connect(handle_recognized)
        recognizer.session_stopped.connect(handle_session_stopped)
        recognizer.canceled.connect(handle_canceled)
        
        # Start recognition
        print("Starting recognition...")
        recognizer.start_continuous_recognition()
        
        # Wait for completion (simple wait)
        import time
        timeout = 300  # 5 minutes max
        start_time = time.time()
        
        while not recognition_done and (time.time() - start_time) < timeout:
            time.sleep(1)
        
        recognizer.stop_continuous_recognition()
        
        # Return result
        full_transcript = " ".join(transcribed_parts).strip()
        
        if full_transcript:
            print(f"Success! Transcript length: {len(full_transcript)} characters")
            return full_transcript
        else:
            print("No speech detected")
            return None
            
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        return None

print("✅ Simple audio transcription function loaded")
