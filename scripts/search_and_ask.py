#!/usr/bin/env python3
"""Search the web, then ask GPT-OSS-20B to analyze the results."""

import sys
import json
import urllib.request
import urllib.parse

VLLM_URL = "http://localhost:8000/v1/chat/completions"

def search_web(query):
    url = "https://api.duckduckgo.com/?q=" + urllib.parse.quote(query) + "&format=json&no_html=1"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PhysLLM/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            results = []
            if data.get("Abstract"):
                results.append("Summary: " + data["Abstract"])
            if data.get("RelatedTopics"):
                for topic in data["RelatedTopics"][:5]:
                    if isinstance(topic, dict) and "Text" in topic:
                        results.append(topic["Text"])
            return "\n".join(results) if results else "No results found for: " + query
    except Exception as e:
        return "Search failed: " + str(e)

def ask_llm(question, context=""):
    if context:
        prompt = "Based on the following search results, answer the question thoroughly.\n\nSEARCH RESULTS:\n" + context + "\n\nQUESTION: " + question
    else:
        prompt = question

    payload = json.dumps({
        "model": "/models/gpt-oss-20b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000
    }).encode()

    req = urllib.request.Request(VLLM_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
        msg = data["choices"][0]["message"]
        return msg["content"], msg.get("reasoning_content", ""), data["usage"]["completion_tokens"]

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 search_and_ask.py 'your question'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print("\n🔍 Searching: " + query)
    print("-" * 50)

    context = search_web(query)
    print("📄 Found " + str(len(context.splitlines())) + " results\n")

    print("🤖 Asking GPT-OSS-20B...")
    print("-" * 50)

    answer, reasoning, tokens = ask_llm(query, context)
    if reasoning:
        print("\n💭 Reasoning: " + reasoning[:300] + "...")
    print("\n📝 Answer:\n" + answer)
    print("\n⚡ " + str(tokens) + " tokens generated")

if __name__ == "__main__":
    main()
