import requests
import json

WHITE_AGENT_URL = "http://localhost:8001/v1/chat/completions"

def run_kickoff():
    print(f"Triggering White Agent at {WHITE_AGENT_URL}...")
    
    payload = {
        "messages": [
            {"role": "user", "content": "Start Assessment"}
        ]
    }

    try:
        response = requests.post(WHITE_AGENT_URL, json=payload)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        print("\nAssessment Complete!")
        print("Report from White Agent:")
        print("---------------------------------------------------")

        try:
            print(json.dumps(json.loads(content), indent=2))
        except:
            print(content)
        print("---------------------------------------------------")

    except Exception as e:
        print(f"\nError during assessment: {e}")
        if 'response' in locals():
            print(f"Response text: {response.text}")

if __name__ == "__main__":
    run_kickoff()