# Check if bronze layer process completed
print("Bronze layer files:")
import os
if os.path.exists(adls_mnt_directory_path_formatted):
    files = os.listdir(adls_mnt_directory_path_formatted)
    print(f"Files in bronze layer: {len(files)}")
    for file in files:
        print(f"  - {file}")
        if file.lower().endswith('.mp3'):
            print(f"    ✅ MP3 file ready for processing: {file}")
else:
    print("❌ Bronze layer directory not found")
    print(f"Looking for: {adls_mnt_directory_path_formatted}")



# Check audio transcription function
try:
    if 'transcribe_mp3_memory_only' in dir():
        print("✅ Audio function available")
    else:
        print("❌ Need to import audio functions")
        print("Run your audio pipeline notebook first!")
except:
    print("❌ Audio function not available")
