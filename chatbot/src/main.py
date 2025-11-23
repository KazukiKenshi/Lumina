
# mental_health_chatbot.py
from langgraph.graph import StateGraph, END
from typing import Dict, Any
import time
from datetime import datetime
from langgraph.graph import StateGraph, END
from src.db_utils import get_recent_exchanges
from src.nodes import (
    MentalHealthState,
    safety_check_node, mode_select_node, profile_analysis_node, retrieve_node,
    diagnosis_node, generate_node, log_node
)

def build_graph():
    graph = StateGraph(MentalHealthState)
    graph.add_node("safety_check", safety_check_node)
    graph.add_node("mode_select", mode_select_node)
    graph.add_node("profile_analysis", profile_analysis_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("diagnosis", diagnosis_node)
    graph.add_node("generate", generate_node)
    graph.add_node("log", log_node)
    graph.add_edge("safety_check", "mode_select")
    graph.add_edge("mode_select", "profile_analysis")
    graph.add_edge("profile_analysis", "retrieve")
    graph.add_edge("retrieve", "diagnosis")
    graph.add_edge("diagnosis", "generate")
    graph.add_edge("generate", "log")
    graph.add_edge("log", END)
    graph.set_entry_point("safety_check")
    return graph.compile()


class ChatSession:
    def __init__(self, user_id=None):
        self.app = build_graph()
        self.user_id = user_id
        self.state = {
            "conversation_history": get_recent_exchanges(4, user_id),
            "user_profile": {
                "symptom_mentions": {},
                "severity_indicators": [],
                "coping_strategies_tried": [],
                "support_system": None,
                "professional_help": None,
                "mode_history": []
            },
            "diagnosed_conditions": [],
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S")
        }

    def set_user(self, user_id):
        if user_id != self.user_id:
            self.user_id = user_id
            self.state = {
                "conversation_history": get_recent_exchanges(4, user_id),
                "user_profile": {
                    "symptom_mentions": {},
                    "severity_indicators": [],
                    "coping_strategies_tried": [],
                    "support_system": None,
                    "professional_help": None,
                    "mode_history": []
                },
                "diagnosed_conditions": [],
                "session_id": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
    def chat(self, user_input: str) -> Dict[str, str]:
        self.state["user_input"] = user_input
        result = self.app.invoke(self.state)
        self.state = result
        return {
            "text": result["response"],
            "emotion": result.get("expression", "neutral")
        }
    def get_summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.state["session_id"],
            "message_count": len(self.state["conversation_history"]) // 2,
            "diagnosed_conditions": list(set(self.state.get("diagnosed_conditions", []))),
            "symptom_profile": self.state.get("user_profile", {}).get("symptom_mentions", {})
        }

if __name__ == "__main__":
    print("=" * 60)
    print("Mental Health Chatbot with RAG")
    print("Type 'quit' to exit, 'summary' for session summary")
    print("=" * 60)
    session = ChatSession()
    test_mode = input("Run test mode? (y/n): ").lower() == 'y'
    if test_mode:
        test_inputs = [
            "I've been feeling really sad and tired for the past few weeks",
            "I can't sleep at night and I've lost interest in things I used to enjoy",
            "Do you think this could be depression?",
            "What can I do to feel better?",
        ]
        for inp in test_inputs:
            print(f"\n{'='*60}")
            print(f"USER: {inp}")
            response = session.chat(inp)
            print(f"ASSISTANT: {response['text']}")
            print(f"EMOTION: {response['emotion']}")
            time.sleep(1)
        print(f"\n{'='*60}")
        print("SESSION SUMMARY:")
        print(session.get_summary())
    else:
        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                print("\nThank you for talking with me. Take care of yourself.")
                print("\nSession Summary:")
                print(session.get_summary())
                break
            if user_input.lower() == 'summary':
                print("\nSession Summary:")
                print(session.get_summary())
                continue
            response = session.chat(user_input)
            print(f"\nAssistant: {response['text']}")
            print(f"Emotion: {response['emotion']}")