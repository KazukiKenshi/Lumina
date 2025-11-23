
# ---------------------------------
# State Schema with Conversation History
# ---------------------------------
from typing import Dict, Any, Optional, TypedDict, List
import re
import time
from datetime import datetime
from sentence_transformers import util
from src.context import logger, rag_system, matcher, client
from src.post_process import format_for_avatar
from src.mode_classifier import classify_mode_with_context

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

# ---------------------------------
# Safety Classifier with Enhanced Detection
# ---------------------------------
def safety_classifier(user_input: str) -> Dict[str, Any]:
	text = user_input.lower().strip()
	critical_keywords = [
		"suicide", "kill myself", "end my life", "want to die",
		"better off dead", "cut myself", "hurt myself", "overdose"
	]
	high_risk_keywords = [
		"hopeless", "worthless", "no reason to live",
		"can't go on", "give up"
	]
	negations = ["not", "no", "never", "don’t", "dont", "didn’t", "didnt", "wasn’t", "wasnt", "without"]
	for kw in critical_keywords:
		pattern = rf"(?:{'|'.join(negations)})\W+(?:\w+\W+){{0,2}}{re.escape(kw)}"
		if re.search(pattern, text):
			print(f"[SAFETY] Negated keyword detected around '{kw}', marking as safe.")
			return {"safe": True, "mode": "diagnosis", "risk_level": "low"}
	for kw in critical_keywords:
		if kw in text:
			print(f"[SAFETY] Critical risk detected: {kw}")
			return {"safe": False, "mode": "crisis", "risk_level": "critical"}
	for kw in high_risk_keywords:
		if kw in text:
			print(f"[SAFETY] High-risk indicator detected: {kw}")
			return {"safe": True, "mode": "counselling", "risk_level": "high"}
	return {"safe": True, "mode": "general", "risk_level": "low"}

def detect_help_intent(user_input: str) -> bool:
	text = user_input.lower()
	help_phrases = [
		"what should i do", "how can i get better", "how can i cope", "any advice",
		"help me", "how do i handle", "how to manage", "how can i deal",
		"how to feel better", "how to fix", "how to stop", "how do i improve",
		"can you help", "tell me what to do"
	]
	return any(phrase in text for phrase in help_phrases)

def analyze_user_profile(state: Dict[str, Any]) -> Dict[str, Any]:
	if "user_profile" not in state or not state["user_profile"]:
		state["user_profile"] = {
			"symptom_mentions": {},
			"severity_indicators": [],
			"coping_strategies_tried": [],
			"support_system": None,
			"professional_help": None
		}
	if "symptom_mentions" not in state["user_profile"]:
		state["user_profile"]["symptom_mentions"] = {}
	user_input = state["user_input"].lower()
	profile = state["user_profile"]
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
			previous = profile["symptom_mentions"].get(category, 0.8)
			profile["symptom_mentions"][category] = round(previous * 0.9 + 1.0, 2)
		else:
			if category in profile["symptom_mentions"]:
				profile["symptom_mentions"][category] = round(profile["symptom_mentions"][category] * 0.9, 2)
	return state

def analyze_symptoms_for_diagnosis(state: Dict[str, Any]) -> Dict[str, Any]:
	if state.get("mode") != "diagnosis":
		return state
	user_input = state["user_input"]
	user_profile = state.get("user_profile", {})
	conversation_history = state.get("conversation_history", [])
	last_user_messages = [msg["content"] for msg in conversation_history[-6:] if msg["role"] == "user"]
	aggregated_text = " ".join(last_user_messages + [user_input])
	symptoms = user_profile.get("symptom_mentions", {})
	symptom_summary = ", ".join(f"{k} ({v:.1f})" for k, v in symptoms.items() if v > 0.3)
	composite_input = f"Recent user statements: {aggregated_text}\nSymptom history: {symptom_summary or 'none'}"
	matches = matcher.match(composite_input)
	top_matches = [m for m in matches if m[1] > 0.35]
	state["diagnosis_confidence"] = {cond: round(score, 3) for cond, score in top_matches}
	if top_matches:
		best_cond, best_score = top_matches[0]
	else:
		best_cond, best_score = None, 0.0
	if best_score < 0.55:
		state["response"] = (
			"I understand. To get a clearer sense, could you tell me more about how your sleep, energy, or mood have been recently?"
		)
		return state
	diagnosed_conditions = [cond for cond, score in state["diagnosis_confidence"].items() if score >= 0.55]
	if diagnosed_conditions:
		state["diagnosed_conditions"] = list(set(state.get("diagnosed_conditions", []) + diagnosed_conditions))
		print(f"[DIAGNOSIS] Conditions suspected: {diagnosed_conditions}")
		state["mode"] = "counselling"
		state["response"] = (
			f"It seems like your experiences might align with {', '.join(diagnosed_conditions)}. "
			f"While I can't diagnose, these patterns are often linked to that area. "
			"Would you like to talk about coping strategies or ways to manage how you've been feeling?"
		)
		return state
	state["response"] = (
		"Thank you for sharing that. I’m still trying to understand the overall picture. "
		"Could you tell me about when these feelings started or how they affect your day-to-day life?"
	)
	return state

def crisis_response(user_input: str) -> str:
	return """I'm very concerned about your safety right now. Please know that you're not alone, and there is help available.\n\n🆘 **Immediate Action Needed:**\n\n**India Emergency Resources:**\n• AASRA Helpline: 91-9820466726 (24/7)\n• Vandrevala Foundation: 9999 666 555 (24/7)\n• Sneha India: 91-44-24640050\n• iCall: 9152987821 (Mon-Sat, 8am-10pm)\n\n**International:**\n• Find your country's helpline: findahelpline.com\n• International Association for Suicide Prevention: iasp.info\n\n**Right Now, You Can:**\n1. Call one of the helplines above - they are trained to help\n2. Go to your nearest emergency room\n3. Reach out to a trusted friend or family member\n4. Text a crisis line if calling feels too difficult\n\n**Safety Plan:**\n• Remove access to means of self-harm\n• Stay with someone you trust\n• Don't use alcohol or drugs right now\n\nYour life has value. This pain is temporary, even though it doesn't feel that way. Please reach out for help right now. Would you like me to help you think through who you could contact?"""

def call_mistral_with_retry(client, model, messages, max_retries=3):
	fallback_model = "mistral-tiny"
	for i in range(max_retries):
		try:
			return client.chat.complete(model=model, messages=messages)
		except Exception as e:
			err_str = str(e)
			if "service_tier_capacity_exceeded" in err_str or "3505" in err_str:
				print(f"[WARN] {model} is at capacity. Switching to {fallback_model}.")
				try:
					return client.chat.complete(model=fallback_model, messages=messages)
				except Exception as inner_e:
					print(f"[WARN] Fallback also failed: {inner_e}")
					return None
			if "429" in err_str:
				wait = (2 ** i) + 2
				print(f"[WARN] Rate limited, retrying in {wait}s...")
				time.sleep(wait)
				continue
			print(f"[ERROR] Mistral API error: {e}")
			return None
	raise Exception("Max retries exceeded or all fallbacks failed.")

def generate_response(state: Dict[str, Any]) -> str:
	user_input = state["user_input"]
	retrieved_context = state.get("retrieved_context", "")
	mode = state["mode"]
	conversation_history = state.get("conversation_history", [])
	history_text = ""
	if conversation_history:
		recent = conversation_history[-4:]
		for msg in recent:
			role = "User" if msg["role"] == "user" else "Assistant"
			history_text += f"{role}: {msg['content']}\n\n"
	system_prompts = {
		"diagnosis": """You are a mental health assessment assistant with knowledge of DSM-5 criteria.\n\nYour role:\n- Help users understand possible mental health conditions based on their symptoms\n- NEVER make definitive diagnoses - only licensed professionals can diagnose\n- Suggest possible conditions that warrant evaluation\n- Always recommend consulting a mental health professional\n- Be compassionate, clear, and non-alarming\n- Ask follow-up questions about symptom duration, severity, and impact\n\nIMPORTANT: You must respond in JSON format with two fields:\n{\n  "text": "your response text here",\n  "expression": "neutral" | "sad" | "happy"\n}\n\nChoose the expression that best matches the tone of your response:\n- "neutral": For factual information, questions, or general discussion\n- "sad": When showing empathy for difficult feelings or discussing challenging topics\n- "happy": When being encouraging, celebrating progress, or providing hope""",
		"counselling": """You are an empathetic, trained mental health counsellor.\n\nYour approach:\n- Practice active listening and validation\n- Use evidence-based techniques (CBT, mindfulness, ACT)\n- Offer specific, actionable coping strategies\n- Normalize struggles while encouraging growth\n- Be warm, non-judgmental, and supportive\n- Recognize when professional help is needed\n- Follow up on previous discussions\n\nIMPORTANT: You must respond in JSON format with two fields:\n{\n  "text": "your response text here",\n  "expression": "neutral" | "sad" | "happy"\n}\n\nChoose the expression that best matches the tone of your response:\n- "neutral": For factual information, questions, or general discussion\n- "sad": When showing empathy for difficult feelings or discussing challenging topics\n- "happy": When being encouraging, celebrating progress, or providing hope""",
		"wellness": """You are a supportive mental wellness coach.\n\nYour role:\n- Provide general mental health education\n- Share prevention and self-care strategies\n- Discuss lifestyle factors affecting mental health\n- Encourage healthy habits and routines\n- Be positive and empowering\n- Recognize when issues need professional attention\n\nIMPORTANT: You must respond in JSON format with two fields:\n{\n  "text": "your response text here",\n  "expression": "neutral" | "sad" | "happy"\n}\n\nChoose the expression that best matches the tone of your response:\n- "neutral": For factual information, questions, or general discussion\n- "sad": When showing empathy for difficult feelings or discussing challenging topics\n- "happy": When being encouraging, celebrating progress, or providing hope"""
	}
	system_prompt = system_prompts.get(mode, system_prompts["wellness"])
	if state.get("diagnosed_conditions"):
		system_prompt += f"\n\nNote: Based on symptoms discussed, possible conditions to consider: {', '.join(set(state['diagnosed_conditions']))}. Remember to recommend professional evaluation."
	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": f"Knowledge base context:\n{retrieved_context}"},
	]
	if history_text:
		messages.append({"role": "user", "content": f"Previous conversation:\n{history_text}"})
	messages.append({"role": "user", "content": f"Current message: {user_input}"})
	try:
		response = call_mistral_with_retry(
			client, "mistral-small-latest", messages
		)
		raw_response = response.choices[0].message.content.strip()
		print(f"\n[RAW JSON RESPONSE]: {raw_response}\n")
		try:
			import json
			cleaned_response = raw_response
			if cleaned_response.startswith('```'):
				cleaned_response = cleaned_response.split('```')[1]
				if cleaned_response.startswith('json'):
					cleaned_response = cleaned_response[4:]
				if '```' in cleaned_response:
					cleaned_response = cleaned_response.split('```')[0]
			cleaned_response = cleaned_response.strip()
			parsed = json.loads(cleaned_response)
			if isinstance(parsed, dict) and 'text' in parsed and 'expression' in parsed:
				print(f"[PARSED] Text: {parsed['text'][:100]}...")
				print(f"[PARSED] Expression: {parsed['expression']}")
				return parsed
			else:
				return {"text": raw_response, "expression": "neutral"}
		except json.JSONDecodeError:
			return {"text": raw_response, "expression": "neutral"}
	except Exception as e:
		return {
			"text": f"I apologize, but I'm having technical difficulties right now. Please try again in a moment. If you're in crisis, please contact emergency services or a crisis helpline immediately. Error: {e}",
			"expression": "sad"
		}

def safety_check_node(state: Dict[str, Any]) -> Dict[str, Any]:
	result = safety_classifier(state["user_input"])
	state.update(result)
	return state

def mode_select_node(state: Dict[str, Any]) -> Dict[str, Any]:
	if state.get("safe", True) and state.get("mode") != "crisis":
		history = " ".join([
			m["content"] for m in state.get("conversation_history", [])[-3:]
			if m["role"] == "user"
		])
		mode, conf = classify_mode_with_context(state["user_input"], history)
		if detect_help_intent(state["user_input"]):
			print("[MODE] Help intent detected — switching to COUNSELLING.")
			mode = "counselling"
		elif any(w in state["user_input"].lower() for w in ["thanks", "thank you", "okay", "ok", "appreciate"]):
			print("[MODE] Gratitude detected — switching to WELLNESS.")
			mode = "wellness"
		elif state.get("diagnosis_confidence"):
			top_conf = max(state["diagnosis_confidence"].values(), default=0.0)
			if top_conf >= 0.55 and mode == "diagnosis":
				print("[MODE] Diagnosis confidence high — transitioning to COUNSELLING.")
				mode = "counselling"
		state["mode"] = mode
		state["mode_confidence"] = conf
		mode_history = state.get("mode_history", [])
		mode_history.append(mode.upper())
		if len(mode_history) > 5:
			mode_history.pop(0)
		state["mode_history"] = mode_history
		print(f"[MODE] {mode.upper()} (confidence: {conf:.2f})")
	return state

def profile_analysis_node(state: Dict[str, Any]) -> Dict[str, Any]:
	return analyze_user_profile(state)

def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
	if state.get("safe", True):
		mode = state.get("mode", "wellness")
		query = state.get("user_input", "")
		matched_tags = rag_system.match_tags(query, mode)
		tokens = [t for t in re.split(r"\W+", query.lower()) if t]
		unique_tokens = len(set(tokens))
		broad = (len(query) > 80) or (unique_tokens > 14) or (not matched_tags and len(query) > 40)
		k = 5 if broad else 3
		try:
			state["retrieved_context"] = rag_system.retrieve(
				query,
				category=mode,
				k=k,
				hint_conditions=matched_tags or None,
				summarize=True
			)
			print(f"[RETRIEVE] mode={mode} k={k} tags={matched_tags}")
		except Exception as e:
			print(f"[WARN] Retrieval failed: {e}")
			state["retrieved_context"] = ""
	return state

def diagnosis_node(state: Dict[str, Any]) -> Dict[str, Any]:
	return analyze_symptoms_for_diagnosis(state)

def is_repeated_response(new_text, past_texts, threshold=0.85):
	embeddings = [matcher.embedder.embed_query(new_text)] + [matcher.embedder.embed_query(t) for t in past_texts]
	new_emb, past_embs = embeddings[0], embeddings[1:]
	similarities = [util.cos_sim(new_emb, p).item() for p in past_embs]
	return any(s > threshold for s in similarities)

def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
	if not state.get("safe", True):
		state["response"] = crisis_response(state["user_input"])
		state["expression"] = "sad"
	if state.get("mode") == "diagnosis" and state.get("diagnosis_confidence"):
		best_score = max(state["diagnosis_confidence"].values(), default=0.0)
		if best_score >= 0.55:
			state["mode"] = "counselling"
	else:
		response_data = generate_response(state)
		if isinstance(response_data, dict):
			state["response"] = response_data.get("text", str(response_data))
			state["expression"] = response_data.get("expression", "neutral")
		else:
			state["response"] = response_data
			state["expression"] = "neutral"
		if state.get("mode") == "diagnosis" and state.get("diagnosis_confidence"):
			conf_text = []
			for cond, score in state["diagnosis_confidence"].items():
				if score >= 0.4:
					conf_text.append(f"{cond.capitalize()} ({score:.2f})")
			if conf_text:
				confidence_summary = (
					"\n\n---\n**Preliminary Symptom Match:**\n"
					+ ", ".join(conf_text)
					+ "\n*(These are similarity estimates — not medical diagnoses. "
					  "Please consult a licensed professional for assessment.)*"
				)
				state["response"] += confidence_summary
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
	timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	print(f"\n[{timestamp}] Mode: {state.get('mode')} | Risk: {state.get('risk_level', 'N/A')}")
	print(f"User: {state['user_input']}")
	if state.get("diagnosed_conditions"):
		print(f"Tracked conditions: {set(state['diagnosed_conditions'])}")
	if state.get("diagnosis_confidence"):
		print(f"Diagnosis confidence: {state['diagnosis_confidence']}")
	avatar_meta = state.get("avatar_output")
	if avatar_meta:
		print(f"Emotion: {avatar_meta.get('emotion')}")
		print(f"Expression: {avatar_meta.get('expression')}")
		print(f"Animation: {avatar_meta.get('animation')}")
		print(f"Estimated speech duration: {avatar_meta.get('duration')}s")
	try:
		lines = []
		lines.append('-----')
		lines.append(f"Session: {state.get('session_id', 'N/A')}")
		lines.append(f"Timestamp: {timestamp}")
		lines.append(f"Mode: {state.get('mode')} | Risk: {state.get('risk_level', 'N/A')}")
		if state.get('diagnosed_conditions'):
			lines.append(f"Diagnosed(approx): {', '.join(set(state['diagnosed_conditions']))}")
		if state.get('diagnosis_confidence'):
			lines.append(f"Diagnosis confidence: {state['diagnosis_confidence']}")
		lines.append(f"User: {state.get('user_input')}")
		lines.append(f"Assistant: {state.get('response')}")
		if avatar_meta:
			lines.append(f"AvatarEmotion: {avatar_meta.get('emotion')} | Expression: {avatar_meta.get('expression')}")
			lines.append(f"Animation: {avatar_meta.get('animation')} | Duration: {avatar_meta.get('duration')}s")
		logger.info('\n'.join(lines))
	except Exception as e:
		print(f"[LOG ERROR] Failed to write to log file: {e}")
	return state

__all__ = [
	"MentalHealthState",
	"safety_classifier", "detect_help_intent", "analyze_user_profile", "analyze_symptoms_for_diagnosis",
	"crisis_response", "call_mistral_with_retry", "generate_response",
	"safety_check_node", "mode_select_node", "profile_analysis_node", "retrieve_node",
	"diagnosis_node", "generate_node", "log_node"
]
