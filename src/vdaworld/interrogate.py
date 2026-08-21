import argparse
import json
import logging
import os
import pathlib
import sys

from vdaworld.api.vlm import VLMClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Interrogate a previous VLM chat.")
    parser.add_argument("--chat_dir", required=True, type=str, help="Path to the interaction folder (e.g. llm_interactions/0_generation)")
    parser.add_argument("--question", required=True, type=str, help="Follow-up question to ask the model.")
    args = parser.parse_args()

    chat_dir = pathlib.Path(args.chat_dir)
    if not chat_dir.is_dir():
        logger.error(f"Directory not found: {chat_dir}")
        sys.exit(1)

    state_path = chat_dir / "state.json"
    prompt_path = chat_dir / "prompt.md"
    response_path = chat_dir / "response.md"

    if not state_path.exists():
        logger.error(f"Missing state.json in {chat_dir}")
        sys.exit(1)

    with open(state_path, "r") as f:
        state = json.load(f)

    if not prompt_path.exists() or not response_path.exists():
        logger.error(f"Missing prompt.md or response.md in {chat_dir}")
        sys.exit(1)

    with open(prompt_path, "r") as f:
        original_prompt = f.read()

    with open(response_path, "r") as f:
        original_response = f.read()

    thinking_path = chat_dir / "thinking.md"
    original_thinking = ""
    if thinking_path.exists():
        with open(thinking_path, "r") as f:
            original_thinking = f.read()

    # We will instantiate the VLMClient and ask a follow-up
    # We provide the original prompt and response as context
    model_name = state.get("model_name", "gemini-3-flash-preview")
    temperature = state.get("temperature", 0.0)
    image_paths = state.get("image_paths", [])

    client = VLMClient(model_name=model_name, temperature=temperature)

    follow_up_prompt = (
        "Here is a previous interaction.\n\n"
        "--- ORIGINAL PROMPT ---\n"
        f"{original_prompt}\n\n"
    )

    if original_thinking:
        follow_up_prompt += (
            "--- ORIGINAL INTERNAL THOUGHTS ---\n"
            f"{original_thinking}\n\n"
        )

    follow_up_prompt += (
        "--- ORIGINAL RESPONSE ---\n"
        f"{original_response}\n\n"
        "--- FOLLOW-UP QUESTION ---\n"
        f"{args.question}\n\n"
        "Please answer the follow-up question based on the context above. Consider your original internal thoughts when explaining your reasoning."
    )

    logger.info("Sending follow-up question to VLM...")
    reply = client.generate_reply(
        prompt=follow_up_prompt,
        image_paths=image_paths,
        use_tools=False
    )

    print("\n" + "="*40)
    print("VLM Response:")
    print("="*40)
    print(reply)
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
