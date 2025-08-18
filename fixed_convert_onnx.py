import os
path = "/Workspace/Users/anuj.b.sharma@unilever.com/1/Volumes/bnlwe_ai_foundation_rag_dev/unvsg2__/unvsg2_audio_rag_test_raw/2025/08/14/17/38/22/0_sp/"
print(f"Directory exists: {os.path.exists(path)}")
if os.path.exists(path):
    files = os.listdir(path)
    print(f"Total files: {len(files)}")
    for file in files:
        print(f"  - {file}")
        if file.lower().endswith('.mp3'):
            print(f"    ✅ MP3 file found!")
else:
    print("❌ Directory not found")
