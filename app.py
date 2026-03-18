import requests
import streamlit as st
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import json

API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
CHATS_DIR = Path("chats")

st.set_page_config(page_title="My AI Chat", layout="wide")


def load_hf_token() -> str | None:
	try:
		token = st.secrets["HF_TOKEN"]
	except Exception:
		return None

	token = str(token).strip()
	return token if token else None


def send_message(token: str, messages: list) -> str:
	headers = {"Authorization": f"Bearer {token}"}
	payload = {
		"model": MODEL,
		"messages": messages,
		"max_tokens": 512,
	}

	response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
	response.raise_for_status()
	data = response.json()
	return data["choices"][0]["message"]["content"]


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
	
	# Send message to API with full history
	with st.spinner("Sending message..."):
		try:
			assistant_response = send_message(hf_token, active_chat["messages"])
			
			# Add assistant response to history
			active_chat["messages"].append({"role": "assistant", "content": assistant_response})
			save_chat_to_disk(active_chat)
			
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

