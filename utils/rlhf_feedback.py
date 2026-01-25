from services.chat_history_service import ChatHistoryService
import json
import os

def apply_rlhf_feedback(user_feedback, user_id, folder_dir: str = "RLHF_Data"):
    chat_service = ChatHistoryService(user_id=user_id)
    last_message = chat_service.get_last_message()
    
    feedback_data = {
        "last_message": last_message,
        "user_feedback": user_feedback,
        "user_id": user_id
    }
    
    feedback_json = json.dumps(feedback_data)
    files = os.listdir(folder_dir)
    path = os.path.join(folder_dir, f"feedback_{len(files)}.json")
    
    with open(path, "w") as json_file:
        json_file.write(feedback_json)