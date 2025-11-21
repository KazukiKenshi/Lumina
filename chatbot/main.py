# mental_health_chatbot.py
from langgraph.graph import StateGraph, END
from typing import Dict, Any, Optional, TypedDict, List
import os
import re
from dotenv import load_dotenv
from mistralai import Mistral
import time
from datetime import datetime

from rag_store import RAGDataStore
from symptom_reasoning import SymptomMatcher
from post_process import format_for_avatar
from mode_classifier import classify_mode_with_context

from sentence_transformers import util


rag = RAGDataStore()
rag.build_store("diagnosis")
rag.build_store("counselling")
rag.build_store("wellness")

matcher = SymptomMatcher()

load_dotenv()

# Initialize Mistral client
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# ---------------------------------
# State Schema with Conversation History
# ---------------------------------
class MentalHealthState(TypedDict, total=False):
    user_input: str
    safe: bool
    mode: str
    retrieved_context: Optional[str]
    response: Optional[str]
    expression: Optional[str]
    conversation_history: List[Dict[str, str]]
    user_profile: Dict[str, Any]
    session_id: str
    diagnosed_conditions: List[str]


# Initialize RAG system
rag_system = RAGDataStore(base_dir="data", store_dir="vectorstores")



# ---------------------------------
# Safety Classifier with Enhanced Detection
# ---------------------------------

def safety_classifier(user_input: str) -> Dict[str, Any]:
    """
    Context-aware safety detection with negation handling.
    Detects suicidal ideation, self-harm, or hopelessness,
    but avoids triggering on negated statements like
    "I'm not thinking about suicide."
    """

    text = user_input.lower().strip()

    # Define risk keyword groups
    critical_keywords = [
        "suicide", "kill myself", "end my life", "want to die",
        "better off dead", "cut myself", "hurt myself", "overdose"
    ]

    high_risk_keywords = [
        "hopeless", "worthless", "no reason to live",
        "can't go on", "give up"
    ]

    # Common negation words
    negations = ["not", "no", "never", "don’t", "dont", "didn’t", "didnt", "wasn’t", "wasnt", "without"]

    # --- Step 1: detect explicit negations ---
    for kw in critical_keywords:
        # e.g., “not thinking about suicide” → skip
        pattern = rf"(?:{'|'.join(negations)})\W+(?:\w+\W+){{0,2}}{re.escape(kw)}"
        if re.search(pattern, text):
            print(f"[SAFETY] Negated keyword detected around '{kw}', marking as safe.")
            return {"safe": True, "mode": "diagnosis", "risk_level": "low"}

    # --- Step 2: detect actual critical intent ---
    for kw in critical_keywords:
        if kw in text:
            print(f"[SAFETY] Critical risk detected: {kw}")
            return {"safe": False, "mode": "crisis", "risk_level": "critical"}

    # --- Step 3: detect emotional hopelessness ---
    for kw in high_risk_keywords:
        if kw in text:
            print(f"[SAFETY] High-risk indicator detected: {kw}")
            return {"safe": True, "mode": "counselling", "risk_level": "high"}

    # --- Step 4: mild distress or safe ---
    return {"safe": True, "mode": "general", "risk_level": "low"}


def detect_help_intent(user_input: str) -> bool:
    """
    Detect if the user is asking for help, coping advice, or improvement steps.
    Used to trigger transition from diagnosis to counselling.
    """
    text = user_input.lower()
    help_phrases = [
        "what should i do", "how can i get better", "how can i cope", "any advice",
        "help me", "how do i handle", "how to manage", "how can i deal",
        "how to feel better", "how to fix", "how to stop", "how do i improve",
        "can you help", "tell me what to do"
    ]
    return any(phrase in text for phrase in help_phrases)


# ---------------------------------
# User Profile Analysis
# ---------------------------------
def analyze_user_profile(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze conversation to build user profile"""
    
    if "user_profile" not in state or not state["user_profile"]:
        state["user_profile"] = {
            "symptom_mentions": {},
            "severity_indicators": [],
            "coping_strategies_tried": [],
            "support_system": None,
            "professional_help": None
        }
    
    # Ensure symptom_mentions exists
    if "symptom_mentions" not in state["user_profile"]:
        state["user_profile"]["symptom_mentions"] = {}
    
    user_input = state["user_input"].lower()
    profile = state["user_profile"]
    
    # Track symptom mentions
    symptom_keywords = {
        "sleep": ["sleep", "insomnia", "can't sleep", "nightmares"],
        "mood": ["sad", "depressed", "hopeless", "empty", "numb"],
        "anxiety": ["anxious", "worried", "panic", "nervous", "fear"],
        "energy": ["tired", "fatigue", "exhausted", "no energy"],
        "concentration": ["can't focus", "distracted", "forgetful"],
        "social": ["isolated", "alone", "withdrawn", "avoid people"],
        "physical": ["headache", "chest pain", "trembling", "sweating"]
    }
    
    # Weighted symptom memory (recent mentions count more)
    for category, keywords in symptom_keywords.items():
        if any(kw in user_input for kw in keywords):
            previous = profile["symptom_mentions"].get(category, 0.8)
            profile["symptom_mentions"][category] = round(previous * 0.9 + 1.0, 2)
        else:
            # Gradually decay old symptom weight
            if category in profile["symptom_mentions"]:
                profile["symptom_mentions"][category] = round(profile["symptom_mentions"][category] * 0.9, 2)


    return state


# ---------------------------------
# Enhanced Diagnosis Analysis
# ---------------------------------

def analyze_symptoms_for_diagnosis(state: Dict[str, Any]) -> Dict[str, Any]:
    """Multi-turn reasoning for diagnosis using accumulated symptoms and semantic similarity."""

    # Skip if already diagnosed and confident enough
    if state.get("mode") != "diagnosis":
        return state

    user_input = state["user_input"]
    user_profile = state.get("user_profile", {})
    conversation_history = state.get("conversation_history", [])

    # Combine context
    last_user_messages = [msg["content"] for msg in conversation_history[-6:] if msg["role"] == "user"]
    aggregated_text = " ".join(last_user_messages + [user_input])

    symptoms = user_profile.get("symptom_mentions", {})
    symptom_summary = ", ".join(f"{k} ({v:.1f})" for k, v in symptoms.items() if v > 0.3)
    composite_input = f"Recent user statements: {aggregated_text}\nSymptom history: {symptom_summary or 'none'}"

    # Semantic matching
    matches = matcher.match(composite_input)
    top_matches = [m for m in matches if m[1] > 0.35]
    state["diagnosis_confidence"] = {cond: round(score, 3) for cond, score in top_matches}

    # Compute best confidence and condition
    if top_matches:
        best_cond, best_score = top_matches[0]
    else:
        best_cond, best_score = None, 0.0

    # 🧠 If confidence is low — ask for more info
    if best_score < 0.55:
        state["response"] = (
            "I understand. To get a clearer sense, could you tell me more about how your sleep, energy, or mood have been recently?"
        )
        return state

    # 🩺 When confidence crosses threshold — finalize diagnosis
    diagnosed_conditions = [cond for cond, score in state["diagnosis_confidence"].items() if score >= 0.55]
    if diagnosed_conditions:
        state["diagnosed_conditions"] = list(set(state.get("diagnosed_conditions", []) + diagnosed_conditions))
        print(f"[DIAGNOSIS] Conditions suspected: {diagnosed_conditions}")

        # Move to counselling phase automatically
        state["mode"] = "counselling"
        state["response"] = (
            f"It seems like your experiences might align with {', '.join(diagnosed_conditions)}. "
            f"While I can't diagnose, these patterns are often linked to that area. "
            "Would you like to talk about coping strategies or ways to manage how you've been feeling?"
        )
        return state

    # If still uncertain
    state["response"] = (
        "Thank you for sharing that. I’m still trying to understand the overall picture. "
        "Could you tell me about when these feelings started or how they affect your day-to-day life?"
    )
    return state


# ---------------------------------
# Crisis Response
# ---------------------------------
def crisis_response(user_input: str) -> str:
    """Immediate crisis intervention response"""
    return """I'm very concerned about your safety right now. Please know that you're not alone, and there is help available.

🆘 **Immediate Action Needed:**

**India Emergency Resources:**
• AASRA Helpline: 91-9820466726 (24/7)
• Vandrevala Foundation: 9999 666 555 (24/7)
• Sneha India: 91-44-24640050
• iCall: 9152987821 (Mon-Sat, 8am-10pm)

**International:**
• Find your country's helpline: findahelpline.com
• International Association for Suicide Prevention: iasp.info

**Right Now, You Can:**
1. Call one of the helplines above - they are trained to help
2. Go to your nearest emergency room
3. Reach out to a trusted friend or family member
4. Text a crisis line if calling feels too difficult

**Safety Plan:**
• Remove access to means of self-harm
• Stay with someone you trust
• Don't use alcohol or drugs right now

Your life has value. This pain is temporary, even though it doesn't feel that way. Please reach out for help right now. Would you like me to help you think through who you could contact?"""


# ---------------------------------
# Mistral Chat with Retry
# ---------------------------------
def call_mistral_with_retry(client, model, messages, max_retries=3):
    """Handle rate limits and capacity errors with fallbacks."""
    fallback_model = "mistral-tiny"  # much less likely to exceed capacity

    for i in range(max_retries):
        try:
            return client.chat.complete(model=model, messages=messages)
        except Exception as e:
            err_str = str(e)

            # Explicit capacity exceeded handling
            if "service_tier_capacity_exceeded" in err_str or "3505" in err_str:
                print(f"[WARN] {model} is at capacity. Switching to {fallback_model}.")
                try:
                    return client.chat.complete(model=fallback_model, messages=messages)
                except Exception as inner_e:
                    print(f"[WARN] Fallback also failed: {inner_e}")
                    return None

            # Standard 429 retry
            if "429" in err_str:
                wait = (2 ** i) + 2
                print(f"[WARN] Rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue

            # Other failures
            print(f"[ERROR] Mistral API error: {e}")
            return None

    raise Exception("Max retries exceeded or all fallbacks failed.")



# ---------------------------------
# Response Generator
# ---------------------------------
def generate_response(state: Dict[str, Any]) -> str:
    """Generate contextual response using Mistral and RAG"""
    
    user_input = state["user_input"]
    retrieved_context = state.get("retrieved_context", "")
    mode = state["mode"]
    conversation_history = state.get("conversation_history", [])
    
    # Build conversation context
    history_text = ""
    if conversation_history:
        recent = conversation_history[-4:]  # Last 2 exchanges
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n\n"
    
    # Mode-specific system prompts
    system_prompts = {
        "diagnosis": """You are a mental health assessment assistant with knowledge of DSM-5 criteria.

Your role:
- Help users understand possible mental health conditions based on their symptoms
- NEVER make definitive diagnoses - only licensed professionals can diagnose
- Suggest possible conditions that warrant evaluation
- Always recommend consulting a mental health professional
- Be compassionate, clear, and non-alarming
- Ask follow-up questions about symptom duration, severity, and impact

IMPORTANT: You must respond in JSON format with two fields:
{
  "text": "your response text here",
  "expression": "neutral" | "sad" | "happy"
}

Choose the expression that best matches the tone of your response:
- "neutral": For factual information, questions, or general discussion
- "sad": When showing empathy for difficult feelings or discussing challenging topics
- "happy": When being encouraging, celebrating progress, or providing hope""",
        
        "counselling": """You are an empathetic, trained mental health counsellor.

Your approach:
- Practice active listening and validation
- Use evidence-based techniques (CBT, mindfulness, ACT)
- Offer specific, actionable coping strategies
- Normalize struggles while encouraging growth
- Be warm, non-judgmental, and supportive
- Recognize when professional help is needed
- Follow up on previous discussions

IMPORTANT: You must respond in JSON format with two fields:
{
  "text": "your response text here",
  "expression": "neutral" | "sad" | "happy"
}

Choose the expression that best matches the tone of your response:
- "neutral": For factual information, questions, or general discussion
- "sad": When showing empathy for difficult feelings or discussing challenging topics
- "happy": When being encouraging, celebrating progress, or providing hope""",
        
        "wellness": """You are a supportive mental wellness coach.

Your role:
- Provide general mental health education
- Share prevention and self-care strategies
- Discuss lifestyle factors affecting mental health
- Encourage healthy habits and routines
- Be positive and empowering
- Recognize when issues need professional attention

IMPORTANT: You must respond in JSON format with two fields:
{
  "text": "your response text here",
  "expression": "neutral" | "sad" | "happy"
}

Choose the expression that best matches the tone of your response:
- "neutral": For factual information, questions, or general discussion
- "sad": When showing empathy for difficult feelings or discussing challenging topics
- "happy": When being encouraging, celebrating progress, or providing hope"""
    }
    
    system_prompt = system_prompts.get(mode, system_prompts["wellness"])
    
    # Add context about diagnosed conditions
    if state.get("diagnosed_conditions"):
        system_prompt += f"\n\nNote: Based on symptoms discussed, possible conditions to consider: {', '.join(set(state['diagnosed_conditions']))}. Remember to recommend professional evaluation."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Knowledge base context:\n{retrieved_context}"},
    ]
    
    # Add conversation history
    if history_text:
        messages.append({"role": "user", "content": f"Previous conversation:\n{history_text}"})
    
    messages.append({"role": "user", "content": f"Current message: {user_input}"})
    
    try:
        response = call_mistral_with_retry(
            client, "mistral-small-latest", messages
        )
        raw_response = response.choices[0].message.content.strip()
        
        # Print raw JSON for debugging
        print(f"\n[RAW JSON RESPONSE]: {raw_response}\n")
        
        # Try to parse JSON response
        try:
            import json
            # Remove markdown code blocks if present
            cleaned_response = raw_response
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response.split('```')[1]
                if cleaned_response.startswith('json'):
                    cleaned_response = cleaned_response[4:]
                # Clean trailing ``` if present
                if '```' in cleaned_response:
                    cleaned_response = cleaned_response.split('```')[0]
            
            cleaned_response = cleaned_response.strip()
            parsed = json.loads(cleaned_response)
            if isinstance(parsed, dict) and 'text' in parsed and 'expression' in parsed:
                print(f"[PARSED] Text: {parsed['text'][:100]}...")
                print(f"[PARSED] Expression: {parsed['expression']}")
                return parsed
            else:
                # Fallback if JSON doesn't have expected format
                return {"text": raw_response, "expression": "neutral"}
        except json.JSONDecodeError:
            # Fallback if not valid JSON
            return {"text": raw_response, "expression": "neutral"}
            
    except Exception as e:
        return {
            "text": f"I apologize, but I'm having technical difficulties right now. Please try again in a moment. If you're in crisis, please contact emergency services or a crisis helpline immediately. Error: {e}",
            "expression": "sad"
        }


# ---------------------------------
# Graph Nodes
# ---------------------------------
def safety_check_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check for safety concerns"""
    result = safety_classifier(state["user_input"])
    state.update(result)
    return state


def mode_select_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Select mode using local NLP classifier and handle dynamic transitions."""
    if state.get("safe", True) and state.get("mode") != "crisis":
        history = " ".join([
            m["content"] for m in state.get("conversation_history", [])[-3:]
            if m["role"] == "user"
        ])
        mode, conf = classify_mode_with_context(state["user_input"], history)

        # --- Dynamic Mode Transitions ---
        # 1️⃣ If user explicitly asks for help → switch to counselling
        if detect_help_intent(state["user_input"]):
            print("[MODE] Help intent detected — switching to COUNSELLING.")
            mode = "counselling"

        # 2️⃣ If user expresses gratitude → switch to wellness
        elif any(w in state["user_input"].lower() for w in ["thanks", "thank you", "okay", "ok", "appreciate"]):
            print("[MODE] Gratitude detected — switching to WELLNESS.")
            mode = "wellness"

        # 3️⃣ If already diagnosed and confidence is high → prefer counselling
        elif state.get("diagnosis_confidence"):
            top_conf = max(state["diagnosis_confidence"].values(), default=0.0)
            if top_conf >= 0.55 and mode == "diagnosis":
                print("[MODE] Diagnosis confidence high — transitioning to COUNSELLING.")
                mode = "counselling"

        state["mode"] = mode
        state["mode_confidence"] = conf
        # Keep a short memory of modes for smoother transitions
        mode_history = state.get("mode_history", [])
        mode_history.append(mode.upper())
        if len(mode_history) > 5:
            mode_history.pop(0)
        state["mode_history"] = mode_history

        print(f"[MODE] {mode.upper()} (confidence: {conf:.2f})")
    return state




def profile_analysis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build user profile from conversation"""
    return analyze_user_profile(state)


def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve relevant knowledge from external RAG datastore"""
    if state.get("safe", True):
        mode = state.get("mode", "wellness")
        try:
            state["retrieved_context"] = rag_system.retrieve(
                state["user_input"],
                category=mode,
                k=3
            )
        except Exception as e:
            print(f"[WARN] Retrieval failed: {e}")
            state["retrieved_context"] = ""
    return state



def diagnosis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze symptoms for diagnosis mode"""
    return analyze_symptoms_for_diagnosis(state)

def is_repeated_response(new_text, past_texts, threshold=0.85):
    embeddings = [matcher.embedder.embed_query(new_text)] + [matcher.embedder.embed_query(t) for t in past_texts]
    new_emb, past_embs = embeddings[0], embeddings[1:]
    similarities = [util.cos_sim(new_emb, p).item() for p in past_embs]
    return any(s > threshold for s in similarities)

def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the assistant's response, optionally including diagnosis confidence."""
    if not state.get("safe", True):
        # Crisis handling shortcut
        state["response"] = crisis_response(state["user_input"])
        state["expression"] = "sad"
    if state.get("mode") == "diagnosis" and state.get("diagnosis_confidence"):
        best_score = max(state["diagnosis_confidence"].values(), default=0.0)
        if best_score >= 0.55:
            state["mode"] = "counselling"
    else:
        # Generate contextual Mistral response
        response_data = generate_response(state)
        
        # Handle both dict (JSON) and string responses
        if isinstance(response_data, dict):
            state["response"] = response_data.get("text", str(response_data))
            state["expression"] = response_data.get("expression", "neutral")
        else:
            state["response"] = response_data
            state["expression"] = "neutral"

        # --- Add confidence summary if in diagnosis mode ---
        if state.get("mode") == "diagnosis" and state.get("diagnosis_confidence"):
            conf_text = []
            for cond, score in state["diagnosis_confidence"].items():
                if score >= 0.4:  # filter low-confidence noise
                    conf_text.append(f"{cond.capitalize()} ({score:.2f})")
            if conf_text:
                confidence_summary = (
                    "\n\n---\n**Preliminary Symptom Match:**\n"
                    + ", ".join(conf_text)
                    + "\n*(These are similarity estimates — not medical diagnoses. "
                      "Please consult a licensed professional for assessment.)*"
                )
                state["response"] += confidence_summary

    # 🧠 Memory check goes here
    previous_responses = [
        msg["content"].lower() for msg in state.get("conversation_history", [])[-6:]
        if msg["role"] == "assistant"
    ]
    if is_repeated_response(state["response"], previous_responses, threshold=0.85):
        print("[MEMORY] Detected semantically repeated question, rephrasing to avoid redundancy.")
        state["response"] = (
            "Thanks for sharing that earlier. I remember we discussed it — "
            "let's focus on how we can help you cope or make progress from here."
        )
        state["expression"] = "neutral"

    # --- Update conversation history ---
    if "conversation_history" not in state:
        state["conversation_history"] = []

    state["conversation_history"].append({
        "role": "user",
        "content": state["user_input"],
        "timestamp": datetime.now().isoformat()
    })
    state["conversation_history"].append({
        "role": "assistant",
        "content": state["response"],
        "timestamp": datetime.now().isoformat()
    })

    state["avatar_output"] = format_for_avatar(state)
    return state



def log_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Log conversation and avatar emotion metadata."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Mode: {state.get('mode')} | Risk: {state.get('risk_level', 'N/A')}")
    print(f"User: {state['user_input']}")
    # Removed duplicate print - response is printed in interactive mode

    if state.get("diagnosed_conditions"):
        print(f"Tracked conditions: {set(state['diagnosed_conditions'])}")

    if state.get("diagnosis_confidence"):
        print(f"Diagnosis confidence: {state['diagnosis_confidence']}")

    # 🆕 Log emotion + animation metadata if available
    avatar_meta = state.get("avatar_output")
    if avatar_meta:
        print(f"Emotion: {avatar_meta.get('emotion')}")
        print(f"Expression: {avatar_meta.get('expression')}")
        print(f"Animation: {avatar_meta.get('animation')}")
        print(f"Estimated speech duration: {avatar_meta.get('duration')}s")

    return state



# ---------------------------------
# Build the Graph
# ---------------------------------
def build_graph():
    """Construct the LangGraph workflow"""
    graph = StateGraph(MentalHealthState)
    
    # Add nodes
    graph.add_node("safety_check", safety_check_node)
    graph.add_node("mode_select", mode_select_node)
    graph.add_node("profile_analysis", profile_analysis_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("diagnosis", diagnosis_node)
    graph.add_node("generate", generate_node)
    graph.add_node("log", log_node)
    
    # Add edges
    graph.add_edge("safety_check", "mode_select")
    graph.add_edge("mode_select", "profile_analysis")
    graph.add_edge("profile_analysis", "retrieve")
    graph.add_edge("retrieve", "diagnosis")
    graph.add_edge("diagnosis", "generate")
    graph.add_edge("generate", "log")
    graph.add_edge("log", END)
    
    graph.set_entry_point("safety_check")
    
    return graph.compile()


# ---------------------------------
# Interactive Chat Session
# ---------------------------------
class ChatSession:
    def __init__(self):
        self.app = build_graph()
        self.state = {
            "conversation_history": [],
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
        """Process a single user message and return response with emotion"""
        self.state["user_input"] = user_input
        result = self.app.invoke(self.state)
        
        # Update persistent state
        self.state = result
        
        return {
            "text": result["response"],
            "emotion": result.get("expression", "neutral")
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get session summary"""
        return {
            "session_id": self.state["session_id"],
            "message_count": len(self.state["conversation_history"]) // 2,
            "diagnosed_conditions": list(set(self.state.get("diagnosed_conditions", []))),
            "symptom_profile": self.state.get("user_profile", {}).get("symptom_mentions", {})
        }


# ---------------------------------
# Main Execution
# ---------------------------------
if __name__ == "__main__":

    # for domain in ["diagnosis", "counselling", "wellness"]:
    #     path = os.path.join("vectorstores", domain)
    #     if not os.path.exists(path):
    #         print(f"Building vector store for {domain}...")
    #         rag_system.build_store(domain)

    print("=" * 60)
    print("Mental Health Chatbot with RAG")
    print("Type 'quit' to exit, 'summary' for session summary")
    print("=" * 60)
    
    session = ChatSession()
    
    # Test mode
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
            time.sleep(1)  # Avoid rate limiting
        
        print(f"\n{'='*60}")
        print("SESSION SUMMARY:")
        print(session.get_summary())
    
    else:
        # Interactive mode
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