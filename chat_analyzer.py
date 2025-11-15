import os
import json
import argparse
import google.generativeai as genai
from dotenv import load_dotenv

def analyze_chat_history(chat_file_path, chunk_size, start_chunk, num_chunks):
    """
    Analyzes a Gemini CLI chat history JSON file to identify and branch debugging sessions.

    Args:
        chat_file_path (str): The path to the chat history JSON file.
        chunk_size (int): The number of messages to include in each chunk.
        start_chunk (int): The 1-based index of the chunk to start processing from.
        num_chunks (int): The maximum number of chunks to process.
    """
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')

    with open(chat_file_path, 'r') as f:
        chat_data = json.load(f)

    messages = chat_data['messages']
    
    checkpoint_file = "analysis_checkpoint.json"
    is_limited_run = start_chunk is not None or num_chunks is not None
    
    # Determine starting point and initial state
    if not is_limited_run and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        main_branch = checkpoint_data.get('main_branch', [])
        summary_log = checkpoint_data.get('summary_log', [])
        debugging_branches = checkpoint_data.get('debugging_branches', {})
        debugging_session_count = checkpoint_data.get('debugging_session_count', 0)
        is_currently_debugging = checkpoint_data.get('is_currently_debugging', False)
        start_chunk_index = checkpoint_data.get('last_processed_chunk', -1) + 1
        print(f"Resuming full analysis from checkpoint at chunk {start_chunk_index + 1}")
    else:
        main_branch, summary_log, debugging_branches, debugging_session_count, is_currently_debugging = [], [], {}, 0, False
        start_chunk_index = 0
        if start_chunk is not None:
            start_chunk_index = start_chunk - 1
            print(f"Starting fresh run from specified chunk: {start_chunk}")
        else:
            print("Starting fresh run from the beginning.")

    total_chunks = (len(messages) + chunk_size - 1) // chunk_size
    
    end_chunk_index = total_chunks
    if num_chunks is not None:
        end_chunk_index = min(start_chunk_index + num_chunks, total_chunks)
        print(f"Will process {num_chunks} chunk(s), ending at chunk {end_chunk_index}")


    for i in range(start_chunk_index * chunk_size, end_chunk_index * chunk_size, chunk_size):
        if i >= len(messages): break
        current_chunk_index = i // chunk_size
        chunk = messages[i:i + chunk_size]
        
        print(f"Processing chunk {current_chunk_index + 1}/{total_chunks}: Sending to Gemini API...")

        prompt = f"""
        You are a chat analysis agent. Your task is to process a chunk of a conversation with a code assistant.
        The conversation is in JSON format.
        Your goal is to identify "debugging side quests". A debugging side quest starts when a bug is found (e.g., compilation error, runtime error, unexpected behavior) and ends when the bug is fixed.

        For the provided chunk of conversation, please do the following:
        1.  Analyze the messages and determine if a debugging side quest is happening.
        2.  If a debugging side quest is identified:
            -   Provide a brief summary of the bug and the fix.
            -   List the message IDs that are part of this debugging session within the current chunk.
        3.  Provide a brief summary of each message in the chunk for the main conversation log.

        Here is the conversation chunk:
        {json.dumps(chunk, indent=2)}

        Please provide your analysis in the following JSON format, and do not wrap it in markdown:
        {{
            "is_debugging": boolean,
            "debugging_summary": "summary of the bug and fix, or null",
            "debugging_message_ids": ["id1", "id2", ...],
            "chunk_summary": [
                {{"id": "message_id", "summary": "brief summary of the message"}},
                ...
            ]
        }}
        """

        try:
            response = model.generate_content(prompt)
            print(f"Processing chunk {current_chunk_index + 1}/{total_chunks}: Received response from Gemini API.")
            
            cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```")
            
            analysis = json.loads(cleaned_text)
            summary_log.extend(analysis.get("chunk_summary", []))
            
            # Even more robustly process debugging message IDs
            raw_debugging_ids = analysis.get("debugging_message_ids", [])
            processed_debugging_ids = []
            if raw_debugging_ids:
                for item in raw_debugging_ids:
                    if isinstance(item, dict) and 'id' in item:
                        processed_debugging_ids.append(item['id'])
                    elif isinstance(item, str):
                        processed_debugging_ids.append(item)
            debugging_message_ids = set(processed_debugging_ids)

            if debugging_message_ids:
                if not is_currently_debugging:
                    is_currently_debugging = True
                    debugging_session_count += 1
                    main_branch.append({"type": "summary", "content": analysis.get("debugging_summary", "A debugging session occurred.")})
                
                session_key = f"debug_session_{debugging_session_count}"
                if session_key not in debugging_branches:
                    debugging_branches[session_key] = []
                
                for message in chunk:
                    if message['id'] in debugging_message_ids:
                        debugging_branches[session_key].append(message)
                    else:
                        main_branch.append(message)
            else:
                is_currently_debugging = False
                main_branch.extend(chunk)

            # Save checkpoint only on full runs
            if not is_limited_run:
                checkpoint_data = {{
                    'main_branch': main_branch,
                    'summary_log': summary_log,
                    'debugging_branches': debugging_branches,
                    'debugging_session_count': debugging_session_count,
                    'is_currently_debugging': is_currently_debugging,
                    'last_processed_chunk': current_chunk_index
                }}
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                print(f"Processing chunk {current_chunk_index + 1}/{total_chunks}: Successfully processed and checkpointed.")

        except (json.JSONDecodeError, Exception) as e:
            print(f"Error processing chunk {current_chunk_index + 1}/{total_chunks}: {e}")
            print(f"Raw API response for chunk {current_chunk_index + 1}:\n{response.text}")
            return

    # Final processing after loop
    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "main_branch.json"), 'w') as f:
        json.dump(main_branch, f, indent=2)

    for session, messages in debugging_branches.items():
        with open(os.path.join(output_dir, f"{session}.json"), 'w') as f:
            json.dump(messages, f, indent=2)

    with open(os.path.join(output_dir, "summary_log.json"), 'w') as f:
        json.dump(summary_log, f, indent=2)
        
    if os.path.exists(checkpoint_file) and not is_limited_run:
        os.remove(checkpoint_file)

    print(f"Analysis complete. Output written to '{output_dir}' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Gemini chat history to identify and branch debugging sessions.")
    parser.add_argument("chat_file", nargs='?', default="chats/session-2025-11-04T13-26-53585682.json", help="The path to the chat history JSON file.")
    parser.add_argument("--chunk-size", type=int, default=10, help="The number of messages per chunk.")
    parser.add_argument("--start-chunk", type=int, help="The 1-based index of the chunk to start processing from.")
    parser.add_argument("--num-chunks", type=int, help="The maximum number of chunks to process.")
    args = parser.parse_args()

    if not os.path.exists(args.chat_file):
        print(f"Error: Chat file not found at '{args.chat_file}'")
    else:
        analyze_chat_history(args.chat_file, args.chunk_size, args.start_chunk, args.num_chunks)
