# mental_health_chatbot.py
from langgraph.graph import StateGraph, END
from typing import Dict, Any, Optional, TypedDict, List
import os
from dotenv import load_dotenv
from mistralai import Mistral
import time
from datetime import datetime

# For RAG components
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

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
    conversation_history: List[Dict[str, str]]
    user_profile: Dict[str, Any]
    session_id: str
    diagnosed_conditions: List[str]


# ---------------------------------
# RAG Setup with Mental Health Corpus
# ---------------------------------
class MentalHealthRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstores = {}
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize separate vector stores for different mental health domains"""
        
        # Diagnosis knowledge base (DSM-5 style)
        diagnosis_docs = [
            Document(page_content="""
            Major Depressive Disorder (MDD):
            Symptoms include: persistent sad mood, loss of interest in activities, 
            significant weight changes, sleep disturbances (insomnia or hypersomnia),
            fatigue, feelings of worthlessness or guilt, difficulty concentrating,
            recurrent thoughts of death. Symptoms must persist for at least 2 weeks.
            """, metadata={"category": "diagnosis", "condition": "depression"}),
            
            Document(page_content="""
            Generalized Anxiety Disorder (GAD):
            Characterized by excessive worry about various events or activities,
            occurring more days than not for at least 6 months. Symptoms include:
            restlessness, fatigue, difficulty concentrating, irritability,
            muscle tension, sleep disturbances. The anxiety is difficult to control.
            """, metadata={"category": "diagnosis", "condition": "anxiety"}),
            
            Document(page_content="""
            Bipolar Disorder:
            Involves episodes of mania/hypomania and depression. Manic symptoms include:
            elevated mood, increased energy, decreased need for sleep, racing thoughts,
            impulsive behavior, inflated self-esteem. Depressive episodes mirror MDD.
            Type I requires at least one manic episode; Type II involves hypomania.
            """, metadata={"category": "diagnosis", "condition": "bipolar"}),
            
            Document(page_content="""
            Panic Disorder:
            Recurrent unexpected panic attacks with intense fear and physical symptoms:
            palpitations, sweating, trembling, shortness of breath, chest pain,
            nausea, dizziness, fear of losing control or dying. Followed by persistent
            worry about future attacks or behavioral changes to avoid attacks.
            """, metadata={"category": "diagnosis", "condition": "panic"}),
            
            Document(page_content="""
            Social Anxiety Disorder:
            Marked fear of social situations where scrutiny by others may occur.
            Fear of embarrassment, humiliation, or negative evaluation. Social situations
            provoke immediate anxiety, are avoided or endured with distress.
            Symptoms persist for 6+ months and cause significant impairment.
            """, metadata={"category": "diagnosis", "condition": "social_anxiety"}),
            
            Document(page_content="""
            Obsessive-Compulsive Disorder (OCD):
            Presence of obsessions (recurrent intrusive thoughts) and/or compulsions
            (repetitive behaviors or mental acts). Common themes: contamination,
            symmetry, forbidden thoughts. Compulsions are performed to reduce anxiety
            but provide only temporary relief. Time-consuming and cause distress.
            """, metadata={"category": "diagnosis", "condition": "ocd"}),
            
            Document(page_content="""
            Post-Traumatic Stress Disorder (PTSD):
            Develops after exposure to traumatic event. Symptoms include: intrusive
            memories, flashbacks, nightmares, avoidance of trauma reminders,
            negative thoughts and mood, hyperarousal (easily startled, on edge).
            Symptoms persist beyond 1 month and cause significant impairment.
            """, metadata={"category": "diagnosis", "condition": "ptsd"}),
        ]
        
        # Counselling/therapeutic techniques
        counselling_docs = [
            Document(page_content="""
            Cognitive Behavioral Therapy (CBT) for Depression:
            Identify negative thought patterns (cognitive distortions) such as
            all-or-nothing thinking, overgeneralization, mental filtering.
            Challenge these thoughts with evidence. Replace with balanced thoughts.
            Behavioral activation: schedule pleasurable activities, even when unmotivated.
            """, metadata={"category": "counselling", "technique": "cbt"}),
            
            Document(page_content="""
            Anxiety Management Techniques:
            Deep breathing: 4-7-8 technique (inhale 4, hold 7, exhale 8).
            Progressive muscle relaxation: tense and release muscle groups systematically.
            Grounding techniques: 5-4-3-2-1 method (5 things you see, 4 you touch, etc.).
            Exposure therapy: gradual confrontation of feared situations.
            """, metadata={"category": "counselling", "technique": "anxiety_management"}),
            
            Document(page_content="""
            Mindfulness and Acceptance:
            Non-judgmental awareness of present moment. Observe thoughts without
            engaging. Mindful breathing meditation. Body scan exercises.
            Acceptance and Commitment Therapy (ACT): accepting difficult emotions
            while committing to value-based actions. Defusion techniques.
            """, metadata={"category": "counselling", "technique": "mindfulness"}),
            
            Document(page_content="""
            Crisis Intervention and Safety Planning:
            Immediate risk assessment. Identify warning signs and triggers.
            Create safety plan: coping strategies, social supports to contact,
            professionals to call, means restriction. Emphasize that crisis is temporary.
            Always recommend professional help for suicidal ideation.
            """, metadata={"category": "counselling", "technique": "crisis"}),
            
            Document(page_content="""
            Sleep Hygiene for Mental Health:
            Maintain consistent sleep schedule. Create relaxing bedtime routine.
            Avoid screens 1 hour before bed. Keep bedroom cool, dark, and quiet.
            Limit caffeine after 2pm. Exercise regularly but not close to bedtime.
            Address racing thoughts with worry journal before bed.
            """, metadata={"category": "counselling", "technique": "sleep"}),
            
            Document(page_content="""
            Building Emotional Resilience:
            Develop strong social connections. Practice self-compassion.
            Set realistic goals and celebrate small wins. Maintain physical health
            through exercise and nutrition. Learn stress management skills.
            Identify personal strengths and values. Seek meaning and purpose.
            """, metadata={"category": "counselling", "technique": "resilience"}),
        ]
        
        # General wellness
        wellness_docs = [
            Document(page_content="""
            The Connection Between Physical and Mental Health:
            Regular exercise releases endorphins and reduces stress hormones.
            Nutrition affects neurotransmitter production (omega-3s, B vitamins).
            Gut-brain axis influences mood. Adequate sleep is crucial for
            emotional regulation. Chronic stress impacts immune function.
            """, metadata={"category": "wellness", "topic": "mind_body"}),
            
            Document(page_content="""
            Building Healthy Relationships:
            Practice active listening. Set healthy boundaries. Communicate needs
            clearly and assertively. Show empathy and validation. Address conflicts
            constructively. Balance give and take. Recognize toxic relationships.
            Seek support from trusted individuals. Join communities of interest.
            """, metadata={"category": "wellness", "topic": "relationships"}),
            
            Document(page_content="""
            Stress Management in Daily Life:
            Time management: prioritize tasks, break large projects into steps.
            Set realistic expectations. Learn to say no. Take regular breaks.
            Practice self-care activities. Engage in hobbies. Limit news consumption.
            Create work-life boundaries. Use relaxation techniques daily.
            """, metadata={"category": "wellness", "topic": "stress"}),
        ]
        
        # Create vector stores
        all_docs = diagnosis_docs + counselling_docs + wellness_docs
        
        # Main store
        self.vectorstores['main'] = FAISS.from_documents(all_docs, self.embeddings)
        
        # Specialized stores
        self.vectorstores['diagnosis'] = FAISS.from_documents(
            diagnosis_docs, self.embeddings
        )
        self.vectorstores['counselling'] = FAISS.from_documents(
            counselling_docs, self.embeddings
        )
        self.vectorstores['wellness'] = FAISS.from_documents(
            wellness_docs, self.embeddings
        )
    
    def retrieve(self, query: str, mode: str, k: int = 3) -> str:
        """Retrieve relevant documents based on mode"""
        store = self.vectorstores.get(mode, self.vectorstores['main'])
        docs = store.similarity_search(query, k=k)
        
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        return context


# Initialize RAG system
rag_system = MentalHealthRAG()


# ---------------------------------
# Safety Classifier with Enhanced Detection
# ---------------------------------
def safety_classifier(user_input: str) -> Dict[str, Any]:
    """Enhanced safety detection with severity levels"""
    
    critical_keywords = [
        "suicide", "kill myself", "end my life", "want to die",
        "better off dead", "cut myself", "hurt myself", "overdose"
    ]
    
    high_risk_keywords = [
        "hopeless", "no point", "can't go on", "give up",
        "no reason to live", "worthless"
    ]
    
    input_lower = user_input.lower()
    
    # Check for critical risk
    if any(kw in input_lower for kw in critical_keywords):
        return {"safe": False, "mode": "crisis", "risk_level": "critical"}
    
    # Check for high risk
    if any(kw in input_lower for kw in high_risk_keywords):
        return {"safe": True, "mode": "counselling", "risk_level": "high"}
    
    return {"safe": True, "mode": "general", "risk_level": "low"}


# ---------------------------------
# Mode Selector with Mistral
# ---------------------------------
def select_mode_mistral(user_input: str, conversation_history: List[Dict]) -> str:
    """Use Mistral to classify intent with conversation context"""
    
    # Build context from recent history
    history_context = ""
    if conversation_history:
        recent = conversation_history[-3:]  # Last 3 exchanges
        history_context = "Recent conversation:\n"
        for msg in recent:
            history_context += f"{msg['role']}: {msg['content'][:100]}\n"
    
    system_prompt = """You are a mental health assistant classifier. 
    Analyze the user's message and classify into ONE category:
    
    - 'diagnosis': User describes symptoms or asks about mental health conditions
    - 'counselling': User seeks emotional support, coping strategies, or therapy techniques
    - 'wellness': User asks about general mental wellness, lifestyle, or prevention
    
    Consider conversation context. If user previously discussed symptoms and now seeks help,
    classify as 'counselling'. Return only the category name."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{history_context}\n\nCurrent message: {user_input}"}
    ]
    
    try:
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=messages,
            temperature=0
        )
        intent = response.choices[0].message.content.strip().lower()
        
        if intent in ["counselling", "diagnosis", "wellness"]:
            return intent
        return "wellness"
    except Exception as e:
        print(f"[WARN] Mode selector failed: {e}")
        return "wellness"


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
    
    for category, keywords in symptom_keywords.items():
        if any(kw in user_input for kw in keywords):
            profile["symptom_mentions"][category] = \
                profile["symptom_mentions"].get(category, 0) + 1
    
    return state


# ---------------------------------
# Diagnosis Analysis
# ---------------------------------
def analyze_symptoms_for_diagnosis(state: Dict[str, Any]) -> Dict[str, Any]:
    """Use Mistral with RAG context to suggest possible conditions"""
    
    if state["mode"] != "diagnosis":
        return state
    
    # Get relevant diagnostic information
    retrieved_context = state.get("retrieved_context", "")
    
    # Build comprehensive symptom history
    symptom_summary = ""
    if state.get("user_profile"):
        symptoms = state["user_profile"].get("symptom_mentions", {})
        if symptoms:
            symptom_summary = "User has mentioned symptoms related to: " + \
                            ", ".join([f"{k} ({v} times)" for k, v in symptoms.items()])
    
    system_prompt = """You are a mental health assessment assistant.
    Based on the user's symptoms and diagnostic criteria provided, suggest possible conditions
    that warrant professional evaluation. 
    
    IMPORTANT:
    - Do NOT make definitive diagnoses
    - Always recommend professional evaluation
    - Mention that only licensed professionals can diagnose
    - List 1-3 possible conditions that match symptoms
    - Explain why symptoms align with these conditions
    - Be compassionate and non-alarming"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Diagnostic criteria:\n{retrieved_context}"},
        {"role": "user", "content": f"{symptom_summary}\n\nUser's current message: {state['user_input']}"}
    ]
    
    try:
        response = call_mistral_with_retry(
            client, "mistral-small-latest", messages
        )
        
        # Extract mentioned conditions for tracking
        content = response.choices[0].message.content
        conditions = ["depression", "anxiety", "bipolar", "ptsd", "ocd", "panic"]
        diagnosed = [c for c in conditions if c in content.lower()]
        
        if "diagnosed_conditions" not in state:
            state["diagnosed_conditions"] = []
        state["diagnosed_conditions"].extend(diagnosed)
        
    except Exception as e:
        print(f"[ERROR] Diagnosis analysis failed: {e}")
    
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
    """Handle rate limiting with exponential backoff"""
    for i in range(max_retries):
        try:
            response = client.chat.complete(model=model, messages=messages)
            return response
        except Exception as e:
            if "429" in str(e):
                wait = 2 ** i
                print(f"[WARN] Rate limited, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise Exception("Max retries exceeded")


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
- Ask follow-up questions about symptom duration, severity, and impact""",
        
        "counselling": """You are an empathetic, trained mental health counsellor.

Your approach:
- Practice active listening and validation
- Use evidence-based techniques (CBT, mindfulness, ACT)
- Offer specific, actionable coping strategies
- Normalize struggles while encouraging growth
- Be warm, non-judgmental, and supportive
- Recognize when professional help is needed
- Follow up on previous discussions""",
        
        "wellness": """You are a supportive mental wellness coach.

Your role:
- Provide general mental health education
- Share prevention and self-care strategies
- Discuss lifestyle factors affecting mental health
- Encourage healthy habits and routines
- Be positive and empowering
- Recognize when issues need professional attention"""
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
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"I apologize, but I'm having technical difficulties right now. Please try again in a moment. If you're in crisis, please contact emergency services or a crisis helpline immediately. Error: {e}"


# ---------------------------------
# Graph Nodes
# ---------------------------------
def safety_check_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check for safety concerns"""
    result = safety_classifier(state["user_input"])
    state.update(result)
    return state


def mode_select_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Select conversation mode"""
    if state.get("safe", True) and state.get("mode") != "crisis":
        state["mode"] = select_mode_mistral(
            state["user_input"],
            state.get("conversation_history", [])
        )
    return state


def profile_analysis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build user profile from conversation"""
    return analyze_user_profile(state)


def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve relevant knowledge from RAG system"""
    if state.get("safe", True):
        mode = state.get("mode", "wellness")
        state["retrieved_context"] = rag_system.retrieve(
            state["user_input"],
            mode,
            k=3
        )
    return state


def diagnosis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze symptoms for diagnosis mode"""
    return analyze_symptoms_for_diagnosis(state)


def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate response"""
    if not state.get("safe", True):
        state["response"] = crisis_response(state["user_input"])
    else:
        state["response"] = generate_response(state)
    
    # Update conversation history
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
    
    return state


def log_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Log conversation for monitoring"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Mode: {state.get('mode')} | Risk: {state.get('risk_level', 'N/A')}")
    print(f"User: {state['user_input']}")
    print(f"Assistant: {state['response']}\n")
    
    if state.get("diagnosed_conditions"):
        print(f"Tracked conditions: {set(state['diagnosed_conditions'])}")
    
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
                "professional_help": None
            },
            "diagnosed_conditions": [],
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
    
    def chat(self, user_input: str) -> str:
        """Process a single user message"""
        self.state["user_input"] = user_input
        result = self.app.invoke(self.state)
        
        # Update persistent state
        self.state = result
        
        return result["response"]
    
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
            print(f"ASSISTANT: {response}")
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
            print(f"\nAssistant: {response}")