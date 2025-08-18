# Cell 12: Main File Processing Loop with Updated MP3 Processing
for filename in os.listdir(adls_mnt_directory_path_formatted):
    print(f" ********* processing file: {filename} **************** ")
    
    if filename.lower().endswith(".pdf"):
        # PDF processing (existing code same...)
        start_time = datetime.now()
        if os.path.getsize(os.path.join(adls_mnt_directory_path_formatted, filename)) > 0:
            filename, documents = parse_pdf_from_document_intelligence(os.path.join(adls_mnt_directory_path_formatted, filename))
            
            # Language detection and translation logic
            language_translation_check_text = documents[0].page_content + documents[-1].page_content 
            language_detection_response = azure_ai_service_language_detection(client, language_translation_check_text)
            source_language_shorthand = language_detection_response["iso6391_name"]

            if source_language_shorthand != "en":
                parse_pdf_response = prepare_pdf_df(filename, documents, translate=True, source_language_shorthand=source_language_shorthand)
            else:
                parse_pdf_response = prepare_pdf_df(filename, documents, translate=False, source_language_shorthand=None)

            end_time = datetime.now()
            if parse_pdf_response["status_code"] == 200:
                parse_pdf_response["dataframe"] = parse_pdf_response["dataframe"].withColumn("page_number", F.col("page_number").cast(IntegerType()))
                all_dfs.append(parse_pdf_response["dataframe"])
                log_metadata.append({
                    "job_id": job_id,
                    "run_id": run_id,
                    "rag_app_name": rag_app_name_from_metadata_table,
                    "rag_app_source_id": rag_app_source_id_from_metadata_table,
                    "source_doc_path": os.path.join(adls_mnt_directory_path_from_metadata_table, filename),
                    "source_doc_name": filename,
                    "target_stage": "silver layer",
                    "target_path": rag_storage_acc_mount_point_from_metadata_table + silver_layer_target_path_from_metadata_table + "created_date=" + str(datetime.now().year) + "-" + str(datetime.now().month) + "-" + str(datetime.now().day),
                    "ingestion_status": "success",
                    "start_time": start_time,
                    "end_time": end_time
                })
            else:
                log_metadata.append({
                    "job_id": job_id,
                    "run_id": run_id,
                    "rag_app_name": rag_app_name_from_metadata_table,
                    "rag_app_source_id": rag_app_source_id_from_metadata_table,
                    "source_doc_path": os.path.join(adls_mnt_directory_path_from_metadata_table, filename),
                    "source_doc_name": filename,
                    "target_stage": "silver layer",
                    "target_path": rag_storage_acc_mount_point_from_metadata_table + silver_layer_target_path_from_metadata_table + "created_date=" + str(datetime.now().year) + "-" + str(datetime.now().month) + "-" + str(datetime.now().day),
                    "ingestion_status": "failure",
                    "error_details": parse_pdf_response["status_details"],
                    "start_time": start_time,
                    "end_time": end_time
                })

    elif filename.lower().endswith(".mp3"):
        # MP3 processing - UPDATED WITH SIMPLE FUNCTION
        print(f"Processing MP3 file: {filename}")
        start_time = datetime.now()
        
        if os.path.getsize(os.path.join(adls_mnt_directory_path_formatted, filename)) > 0:
            try:
                mp3_file_path = os.path.join(adls_mnt_directory_path_formatted, filename)
                
                # Use simple transcription function (NO librosa/numpy issues)
                transcript = transcribe_mp3_simple(mp3_file_path)
                
                if transcript:
                    # Create simple DataFrame with filename and transcript
                    audio_data = [{
                        "file_path": filename,
                        "content": transcript,
                        "row_number": None,
                        "page_number": None,
                        "sheet_name": None,
                        "header": None,
                        "footer": None
                    }]
                    
                    audio_df = spark.createDataFrame(audio_data)
                    all_dfs.append(audio_df)
                    
                    end_time = datetime.now()
                    log_metadata.append({
                        "job_id": job_id,
                        "run_id": run_id,
                        "rag_app_name": rag_app_name_from_metadata_table,
                        "rag_app_source_id": rag_app_source_id_from_metadata_table,
                        "source_doc_path": mp3_file_path,
                        "source_doc_name": filename,
                        "target_stage": "silver layer",
                        "target_path": rag_storage_acc_mount_point_from_metadata_table + silver_layer_target_path_from_metadata_table + "created_date=" + str(datetime.now().year) + "-" + str(datetime.now().month) + "-" + str(datetime.now().day),
                        "ingestion_status": "success",
                        "start_time": start_time,
                        "end_time": end_time
                    })
                    print(f"Successfully processed MP3: {filename}")
                else:
                    raise Exception("No transcription generated")
                    
            except Exception as e:
                end_time = datetime.now()
                print(f"Error processing MP3 {filename}: {str(e)}")
                log_metadata.append({
                    "job_id": job_id,
                    "run_id": run_id,
                    "rag_app_name": rag_app_name_from_metadata_table,
                    "rag_app_source_id": rag_app_source_id_from_metadata_table,
                    "source_doc_path": os.path.join(adls_mnt_directory_path_from_metadata_table, filename),
                    "source_doc_name": filename,
                    "target_stage": "silver layer",
                    "target_path": rag_storage_acc_mount_point_from_metadata_table + silver_layer_target_path_from_metadata_table + "created_date=" + str(datetime.now().year) + "-" + str(datetime.now().month) + "-" + str(datetime.now().day),
                    "ingestion_status": "failure",
                    "error_details": str(e),
                    "start_time": start_time,
                    "end_time": end_time
                })
        else:
            print(f"MP3 file {filename} is empty or corrupt")

    elif filename.lower().endswith(".docx"):
        # DOCX processing (existing code same...)
        start_time = datetime.now()
        if os.path.getsize(os.path.join(adls_mnt_directory_path_formatted, filename)) > 0:
            filename, documents = parse_word_or_image_from_document_intelligence(os.path.join(adls_mnt_directory_path_formatted, filename))
            language_detection_response = azure_ai_service_language_detection(client, documents[0].page_content[:5000])
            source_language_shorthand = language_detection_response["iso6391_name"]

            if source_language_shorthand != "en":
                parse_word_response = prepare_word_or_image_df(filename, documents, translate=True, source_language_shorthand=source_language_shorthand)
            else:
                parse_word_response = prepare_word_or_image_df(filename, documents, translate=False, source_language_shorthand=None)
            
            end_time = datetime.now()
            if parse_word_response["status_code"] == 200:
                all_dfs.append(parse_word_response["dataframe"])
                log_metadata.append({
                    "job_id": job_id,
                    "run_id": run_id,
                    "rag_app_name": rag_app_name_from_metadata_table,
                    "rag_app_source_id": rag_app_source_id_from_metadata_table,
                    "source_doc_path": os.path.join(adls_mnt_directory_path_from_metadata_table, filename),
                    "source_doc_name": filename,
                    "target_stage": "silver layer",
                    "target_path": rag_storage_acc_mount_point_from_metadata_table + silver_layer_target_path_from_metadata_table + "created_date=" + str(datetime.now().year) + "-" + str(datetime.now().month) + "-" + str(datetime.now().day),
                    "ingestion_status": "success",
                    "start_time": start_time,
                    "end_time": end_time
                })
            else:
                log_metadata.append({
                    "job_id": job_id,
                    "run_id": run_id,
                    "rag_app_name": rag_app_name_from_metadata_table,
                    "rag_app_source_id": rag_app_source_id_from_metadata_table,
                    "source_doc_path": os.path.join(adls_mnt_directory_path_from_metadata_table, filename),
                    "source_doc_name": filename,
                    "target_stage": "silver layer",
                    "target_path": rag_storage_acc_mount_point_from_metadata_table + silver_layer_target_path_from_metadata_table + "created_date=" + str(datetime.now().year) + "-" + str(datetime.now().month) + "-" + str(datetime.now().day),
                    "ingestion_status": "failure",
                    "error_details": parse_word_response["status_details"],
                    "start_time": start_time,
                    "end_time": end_time
                })

    # Add other file types (CSV, Images, HTML, PPTX, XLSX) here if needed...
    
    else:
        # Unsupported file type
        log_metadata.append({
            "job_id": job_id,
            "run_id": run_id,
            "rag_app_name": rag_app_name_from_metadata_table,
            "rag_app_source_id": rag_app_source_id_from_metadata_table,
            "source_doc_path": os.path.join(adls_mnt_directory_path_from_metadata_table, filename),
            "source_doc_name": filename,
            "target_stage": "silver layer",
            "target_path": adls_mnt_directory_path_from_metadata_table,
            "ingestion_status": "failure",
            "error_details": "file not supported",
        })

print(f"Processing completed. Total DataFrames created: {len(all_dfs)}")
