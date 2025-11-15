import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

def analyze_chat_history(chat_file_path):
    """
    Analyzes a Gemini CLI chat history JSON file to identify and branch debugging sessions.

    Args:
        chat_file_path (str): The path to the chat history JSON file.
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
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        main_branch = checkpoint_data.get('main_branch', [])
        summary_log = checkpoint_data.get('summary_log', [])
        debugging_branches = checkpoint_data.get('debugging_branches', {})
        debugging_session_count = checkpoint_data.get('debugging_session_count', 0)
        start_chunk_index = checkpoint_data.get('last_processed_chunk', -1) + 1
        print(f"Resuming from chunk {start_chunk_index + 1}")
    else:
        main_branch = []
        summary_log = []
        debugging_branches = {}
        debugging_session_count = 0
        start_chunk_index = 0

    chunk_size = 10
    total_chunks = (len(messages) + chunk_size - 1) // chunk_size

    for i in range(start_chunk_index * chunk_size, len(messages), chunk_size):
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
            
            # Clean the response text to remove markdown fences
            cleaned_text = response.text
            if cleaned_text.strip().startswith("```json"):
                cleaned_text = cleaned_text.strip()[7:]
            if cleaned_text.strip().endswith("```"):
                cleaned_text = cleaned_text.strip()[:-3]
            
            analysis = json.loads(cleaned_text)

            debugging_message_ids = set(analysis.get("debugging_message_ids", []))

            if analysis.get("is_debugging"):
                debugging_session_count += 1
                session_key = f"debug_session_{{debugging_session_count}}"
                if session_key not in debugging_branches:
                    debugging_branches[session_key] = []
                
                main_branch.append({
                    "type": "summary",
                    "content": analysis.get("debugging_summary")
                })

            for message in chunk:
                if message['id'] in debugging_message_ids:
                    if 'session_key' in locals():
                        debugging_branches[session_key].append(message)
                    else: # case where is_debugging is false but message ids are returned
                        main_branch.append(message)
                else:
                    main_branch.append(message)
            
            summary_log.extend(analysis.get("chunk_summary", []))

            # Save checkpoint
            checkpoint_data = {
                'main_branch': main_branch,
                'summary_log': summary_log,
                'debugging_branches': debugging_branches,
                'debugging_session_count': debugging_session_count,
                'last_processed_chunk': current_chunk_index
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            print(f"Processing chunk {current_chunk_index + 1}/{total_chunks}: Successfully processed and checkpointed.")


        except (json.JSONDecodeError, Exception) as e:
            print(f"Error processing chunk {current_chunk_index + 1}/{total_chunks}: {e}")
            print(f"Raw API response for chunk {current_chunk_index + 1}:\n{response.text}")
            return


    # Create output directory
    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)

    # Write main branch
    with open(os.path.join(output_dir, "main_branch.json"), 'w') as f:
        json.dump(main_branch, f, indent=2)

    # Write debugging branches
    for session, messages in debugging_branches.items():
        with open(os.path.join(output_dir, f"{session}.json"), 'w') as f:
            json.dump(messages, f, indent=2)

    # Write summary log
    with open(os.path.join(output_dir, "summary_log.json"), 'w') as f:
        json.dump(summary_log, f, indent=2)
        
    # Clean up checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print(f"Analysis complete. Output written to '{output_dir}' directory.")

if __name__ == "__main__":
    chat_file = "chats/session-2025-11-04T13-26-53585682.json"
    if not os.path.exists(chat_file):
        print(f"Error: Chat file not found at '{chat_file}'")
    else:
        analyze_chat_history(chat_file)