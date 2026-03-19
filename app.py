import requests
import streamlit as st
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import json
import time

API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
CHATS_DIR = Path("chats")
MEMORY_FILE = Path("memory.json")

st.set_page_config(page_title="My AI Chat", layout="wide")


def load_hf_token() -> str | None:
	try:
		token = st.secrets["HF_TOKEN"]
	except Exception:
		return None

	token = str(token).strip()
	return token if token else None


def load_memory_from_disk() -> dict:
	if not MEMORY_FILE.exists() or MEMORY_FILE.stat().st_size == 0:
		return {}

	try:
		with MEMORY_FILE.open("r", encoding="utf-8") as f:
			memory = json.load(f)
		return memory if isinstance(memory, dict) else {}
	except (json.JSONDecodeError, OSError, TypeError, ValueError):
		return {}


def save_memory_to_disk(memory: dict) -> None:
	with MEMORY_FILE.open("w", encoding="utf-8") as f:
		json.dump(memory, f, ensure_ascii=False, indent=2)


def merge_memory(existing: dict, new_memory: dict) -> dict:
	merged = dict(existing)

	for key, value in new_memory.items():
		if isinstance(value, dict) and isinstance(merged.get(key), dict):
			merged[key] = merge_memory(merged[key], value)
		else:
			merged[key] = value

	return merged


def build_model_messages(messages: list[dict], memory: dict) -> list[dict]:
	if not memory:
		return messages

	memory_json = json.dumps(memory, ensure_ascii=False)
	system_prompt = (
		"You are a helpful AI assistant. Use the saved user memory below to personalize your replies when relevant. "
		"Do not mention the memory store directly unless the user asks.\n\n"
		f"Saved user memory: {memory_json}"
	)
	return [{"role": "system", "content": system_prompt}, *messages]


def parse_json_object(text: str) -> dict:
	clean_text = text.strip()
	if clean_text.startswith("```"):
		clean_text = clean_text.strip("`")
		if clean_text.startswith("json"):
			clean_text = clean_text[4:].strip()

	try:
		parsed = json.loads(clean_text)
		return parsed if isinstance(parsed, dict) else {}
	except json.JSONDecodeError:
		start = clean_text.find("{")
		end = clean_text.rfind("}")
		if start != -1 and end != -1 and start < end:
			try:
				parsed = json.loads(clean_text[start : end + 1])
				return parsed if isinstance(parsed, dict) else {}
			except json.JSONDecodeError:
				return {}
	return {}


def extract_user_memory(token: str, user_message: str) -> dict:
	headers = {"Authorization": f"Bearer {token}"}
	payload = {
		"model": MODEL,
		"messages": [
			{
				"role": "system",
				"content": (
					"Extract any durable user facts or preferences from the user's message. "
					"Return only a JSON object. If there are no useful traits, return {}. "
					"Prefer concise keys such as name, preferred_language, interests, communication_style, favorite_topics, or preferences. "
					"Ignore temporary requests that are not long-term preferences."
				),
			},
			{"role": "user", "content": user_message},
		],
		"max_tokens": 128,
	}

	response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
	response.raise_for_status()
	data = response.json()
	content = data["choices"][0]["message"]["content"]
	return parse_json_object(content)


def stream_message(token: str, messages: list):
	headers = {"Authorization": f"Bearer {token}"}
	payload = {
		"model": MODEL,
		"messages": messages,
		"max_tokens": 512,
		"stream": True,
	}

	with requests.post(API_URL, headers=headers, json=payload, timeout=120, stream=True) as response:
		response.raise_for_status()

		for raw_line in response.iter_lines(decode_unicode=True):
			if not raw_line:
				continue
			if not raw_line.startswith("data:"):
				continue

			data_line = raw_line[5:].strip()
			if not data_line or data_line == "[DONE]":
				continue

			try:
				chunk = json.loads(data_line)
			except json.JSONDecodeError:
				continue

			choices = chunk.get("choices", [])
			if not choices:
				continue

			choice = choices[0]
			delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
			content = delta.get("content") if isinstance(delta, dict) else None

			if content is None and isinstance(choice, dict):
				message = choice.get("message", {})
				if isinstance(message, dict):
					content = message.get("content")

			if isinstance(content, str) and content:
				yield content
				time.sleep(0.02)
			elif isinstance(content, list):
				for part in content:
					if isinstance(part, dict):
						text = part.get("text")
						if isinstance(text, str) and text:
							yield text
							time.sleep(0.02)


def now_timestamp() -> str:
	return datetime.now().strftime("%Y-%m-%d %H:%M")


def create_chat() -> dict:
	chat_id = str(uuid4())
	return {
		"id": chat_id,
		"title": "New Chat",
		"created_at": now_timestamp(),
		"messages": [],
	}


def chat_file_path(chat_id: str) -> Path:
	return CHATS_DIR / f"{chat_id}.json"


def save_chat_to_disk(chat: dict) -> None:
	CHATS_DIR.mkdir(parents=True, exist_ok=True)
	with chat_file_path(chat["id"]).open("w", encoding="utf-8") as f:
		json.dump(chat, f, ensure_ascii=False, indent=2)


def delete_chat_file(chat_id: str) -> None:
	file_path = chat_file_path(chat_id)
	if file_path.exists():
		file_path.unlink()


def load_chats_from_disk() -> list[dict]:
	CHATS_DIR.mkdir(parents=True, exist_ok=True)
	loaded_chats: list[dict] = []

	for file_path in CHATS_DIR.glob("*.json"):
		try:
			with file_path.open("r", encoding="utf-8") as f:
				chat = json.load(f)

			if not isinstance(chat, dict):
				continue
			if not isinstance(chat.get("id"), str) or not chat["id"].strip():
				continue

			chat.setdefault("title", "New Chat")
			chat.setdefault("created_at", now_timestamp())
			chat.setdefault("messages", [])

			if not isinstance(chat["messages"], list):
				chat["messages"] = []

			clean_messages = []
			for m in chat["messages"]:
				if isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str):
					clean_messages.append({"role": m["role"], "content": m["content"]})
			chat["messages"] = clean_messages

			loaded_chats.append(chat)
		except (json.JSONDecodeError, OSError, TypeError, ValueError):
			continue

	loaded_chats.sort(key=lambda c: (str(c.get("created_at", "")), str(c.get("id", ""))))
	return loaded_chats


def update_chat_title(chat: dict) -> None:
	if chat["title"] == "New Chat":
		first_user_msg = next((m["content"] for m in chat["messages"] if m["role"] == "user"), "")
		if first_user_msg:
			chat["title"] = first_user_msg[:40] + ("..." if len(first_user_msg) > 40 else "")


def get_active_chat() -> dict | None:
	active_id = st.session_state.active_chat_id
	for chat in st.session_state.chats:
		if chat["id"] == active_id:
			return chat
	return None


# Initialize chat session state
if "chats" not in st.session_state:
	st.session_state.chats = load_chats_from_disk()
	if st.session_state.chats:
		st.session_state.active_chat_id = st.session_state.chats[0]["id"]
	else:
		first_chat = create_chat()
		st.session_state.chats = [first_chat]
		st.session_state.active_chat_id = first_chat["id"]
		save_chat_to_disk(first_chat)

if "active_chat_id" not in st.session_state:
	st.session_state.active_chat_id = st.session_state.chats[0]["id"] if st.session_state.chats else None

if "memory" not in st.session_state:
	st.session_state.memory = load_memory_from_disk()

st.title("My AI Chat")

# Sidebar: chat management
st.sidebar.title("Chats")
if st.sidebar.button("+ New Chat", use_container_width=True):
	new_chat = create_chat()
	st.session_state.chats.append(new_chat)
	st.session_state.active_chat_id = new_chat["id"]
	save_chat_to_disk(new_chat)
	st.rerun()

chat_list_container = st.sidebar.container(height=450)
with chat_list_container:
	for chat in st.session_state.chats:
		col_main, col_delete = st.columns([5, 1])
		is_active = chat["id"] == st.session_state.active_chat_id

		with col_main:
			if st.button(
				chat["title"],
				key=f"open_{chat['id']}",
				type="primary" if is_active else "secondary",
				use_container_width=True,
			):
				st.session_state.active_chat_id = chat["id"]
				st.rerun()
			st.caption(chat["created_at"])

		with col_delete:
			if st.button("✕", key=f"delete_{chat['id']}", use_container_width=True):
				deleted_was_active = chat["id"] == st.session_state.active_chat_id
				delete_chat_file(chat["id"])
				st.session_state.chats = [c for c in st.session_state.chats if c["id"] != chat["id"]]

				if deleted_was_active:
					st.session_state.active_chat_id = (
						st.session_state.chats[0]["id"] if st.session_state.chats else None
					)

				st.rerun()

with st.sidebar.expander("User Memory", expanded=False):
	if st.session_state.memory:
		st.json(st.session_state.memory)
	else:
		st.caption("No saved user memory yet.")

	if st.button("Clear Memory", use_container_width=True):
		st.session_state.memory = {}
		save_memory_to_disk(st.session_state.memory)
		st.rerun()

hf_token = load_hf_token()
if not hf_token:
	st.error(
		"Missing or empty HF token. Add HF_TOKEN to .streamlit/secrets.toml (or Streamlit Cloud Secrets)."
	)
	st.stop()

active_chat = get_active_chat()

if active_chat is None:
	st.info("No chats yet. Create one from the sidebar.")
	st.stop()

# Display conversation history
for message in active_chat["messages"]:
	with st.chat_message(message["role"]):
		st.write(message["content"])

# Fixed input bar at the bottom
user_input = st.chat_input("Type your message here...")

if user_input:
	# Add user message to history
	active_chat["messages"].append({"role": "user", "content": user_input})
	update_chat_title(active_chat)
	save_chat_to_disk(active_chat)

	with st.chat_message("user"):
		st.write(user_input)
	
	try:
		model_messages = build_model_messages(active_chat["messages"], st.session_state.memory)

		with st.chat_message("assistant"):
			assistant_response = st.write_stream(stream_message(hf_token, model_messages))

		if not isinstance(assistant_response, str):
			assistant_response = "" if assistant_response is None else str(assistant_response)

		# Add assistant response to history
		active_chat["messages"].append({"role": "assistant", "content": assistant_response})
		save_chat_to_disk(active_chat)

		try:
			extracted_memory = extract_user_memory(hf_token, user_input)
			if extracted_memory:
				st.session_state.memory = merge_memory(st.session_state.memory, extracted_memory)
				save_memory_to_disk(st.session_state.memory)
		except (requests.RequestException, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
			pass

	except requests.HTTPError as http_err:
		status = http_err.response.status_code if http_err.response is not None else "unknown"
		if status == 401:
			st.error("Invalid token (401). Check HF_TOKEN in secrets.")
		elif status == 429:
			st.error("Rate limit hit (429). Please retry shortly.")
		else:
			st.error(f"API request failed (status {status}).")
		detail = http_err.response.text if http_err.response is not None else str(http_err)
		st.caption(detail)
	except requests.RequestException as req_err:
		st.error("Network failure while contacting Hugging Face API.")
		st.caption(str(req_err))
	except (KeyError, IndexError, TypeError, ValueError) as parse_err:
		st.error("Unexpected response format from Hugging Face API.")
		st.caption(str(parse_err))
	
	st.rerun()

