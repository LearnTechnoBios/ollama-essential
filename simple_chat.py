#!/usr/bin/env python3

import ollama

from typing import List, Dict
import json
from datetime import datetime

def chat_with_model(
    model: str = 'gemma2:2b',
    messages: List[Dict[str, str]] = None,
    print_full_response: bool = False
) -> Dict:
    if messages is None:
        messages = [{
            'role': 'user',
            'content': 'Hi'
        }]
    
    try:
        # Make the API call
        response = ollama.chat(
            model=model,
            messages=messages
        )
        
        # Print just the message content
        print("\nModel's response:")
        print(response['message']['content'])
        
        # Optionally print the full response details
        if print_full_response:
            print("\nFull response details:")
            # Convert the ChatResponse object to a dictionary
            response_dict = {
                'model': response.model,
                'created_at': response.created_at,
                'message': response.message,
                'done': response.done,
                'total_duration': response.total_duration,
                'load_duration': response.load_duration,
                'prompt_eval_count': response.prompt_eval_count,
                'prompt_eval_duration': response.prompt_eval_duration,
                'eval_count': response.eval_count,
                'eval_duration': response.eval_duration
            }
            print(json.dumps(response_dict, indent=2, default=str))
            
        return response
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    messages = [
        {
            'role': 'user',
            'content': 'What is the fastest animal on the planet?'
        },
    ]
    
    chat_with_model(
        model='gemma2:2b',
        messages=messages,
        print_full_response=True
    )