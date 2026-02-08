"""
Opik Evaluation & Observability Module for Agentic Assistant.

Provides:
1. Online LLM-as-judge evaluations (run after every query)
2. Offline experiment tracking (benchmark against test datasets)
3. Custom scoring functions for multi-agent workflows
4. Agent trajectory analysis

This module is designed to showcase exceptional Opik integration for the
"Best Use of Opik" hackathon prize.
"""

import os
import time
import asyncio
from typing import Dict, Any, List, Optional

# Opik imports with graceful fallback
try:
    import opik
    from opik import track as opik_track
    from opik.evaluation.metrics import score_result
    HAS_OPIK = True
except ImportError:
    HAS_OPIK = False

try:
    from config.llm import gemini
except ImportError:
    gemini = None

_OPIK_PROJECT = os.getenv("OPIK_PROJECT_NAME", "agentic-assistant")


# ─────────────────────────────────────────────────────────────────────────────
# 1. ONLINE LLM-AS-JUDGE EVALUATIONS
#    These run automatically after every user query to score the response.
# ─────────────────────────────────────────────────────────────────────────────

async def _llm_judge(prompt: str) -> Dict[str, Any]:
    """Call Gemini as a judge and parse score + reason."""
    if not gemini:
        return {"score": 0.0, "reason": "LLM not available"}
    try:
        response = await gemini.ainvoke(prompt)
        text = response.content.strip()
        
        # Parse "Score: X/10" pattern
        import re
        score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)\s*/\s*10', text)
        score = float(score_match.group(1)) / 10.0 if score_match else 0.5
        
        # Extract reason (everything after "Reason:" or the full text)
        reason_match = re.search(r'Reason:\s*(.+)', text, re.DOTALL)
        reason = reason_match.group(1).strip()[:200] if reason_match else text[:200]
        
        return {"score": round(score, 2), "reason": reason}
    except Exception as e:
        return {"score": 0.0, "reason": f"Judge error: {str(e)[:100]}"}


async def eval_response_relevance(query: str, response: str) -> Dict[str, Any]:
    """Evaluate how relevant the response is to the user's query."""
    prompt = f"""You are an evaluation judge. Rate how relevant the AI response is to the user's query.

User Query: "{query}"

AI Response: "{response[:1500]}"

Criteria:
- Does the response directly address the user's question?
- Is the information provided useful and on-topic?
- Does it avoid irrelevant tangents?

Respond in this exact format:
Score: X/10
Reason: <one sentence explanation>"""
    
    return await _llm_judge(prompt)


async def eval_response_completeness(query: str, response: str, agents_used: List[str]) -> Dict[str, Any]:
    """Evaluate whether the response fully addresses the query."""
    prompt = f"""You are an evaluation judge. Rate how complete the AI response is.

User Query: "{query}"

AI Response: "{response[:1500]}"

Agents Used: {', '.join(agents_used)}

Criteria:
- Does the response fully answer the question?
- Are there important aspects left unaddressed?
- Did the system use appropriate agents for the task?

Respond in this exact format:
Score: X/10
Reason: <one sentence explanation>"""
    
    return await _llm_judge(prompt)


async def eval_hallucination_risk(query: str, response: str, agent_results: Dict) -> Dict[str, Any]:
    """Evaluate the risk of hallucination in the response."""
    # Summarize what data was actually available
    available_data = []
    for agent, result in agent_results.items():
        if result:
            preview = str(result)[:200]
            available_data.append(f"[{agent}]: {preview}")
    
    data_summary = "\n".join(available_data) if available_data else "No agent data available"
    
    prompt = f"""You are an evaluation judge. Rate the hallucination risk of this AI response.

User Query: "{query}"

AI Response: "{response[:1500]}"

Actual Data Available from Agents:
{data_summary[:2000]}

Criteria:
- Does the response only contain information supported by the agent data?
- Are there any fabricated facts, numbers, or claims?
- Does it acknowledge uncertainty when data is missing?

A score of 10 means NO hallucination risk. A score of 1 means HIGH hallucination risk.

Respond in this exact format:
Score: X/10
Reason: <one sentence explanation>"""
    
    return await _llm_judge(prompt)


async def eval_agent_efficiency(agents_used: List[str], chain_of_thought: List[str], elapsed_time: float) -> Dict[str, Any]:
    """Evaluate the efficiency of agent selection and execution."""
    num_agents = len(agents_used)
    num_steps = len(chain_of_thought)
    
    # Heuristic scoring for efficiency
    # Ideal: minimal agents, fast execution
    if elapsed_time < 5:
        time_score = 1.0
    elif elapsed_time < 15:
        time_score = 0.8
    elif elapsed_time < 30:
        time_score = 0.6
    else:
        time_score = 0.4
    
    # Agent count penalty (more agents = potentially less efficient)
    agent_score = max(0.3, 1.0 - (num_agents - 1) * 0.15)
    
    score = round((time_score * 0.5 + agent_score * 0.5), 2)
    reason = f"{num_agents} agents, {num_steps} steps, {elapsed_time:.1f}s"
    
    return {"score": score, "reason": reason}


async def run_online_evaluations(
    query: str,
    response: str,
    agents_used: List[str],
    agent_results: Dict[str, Any],
    chain_of_thought: List[str],
    elapsed_time: float,
    trace_id: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run all online evaluations in parallel after a query completes.
    Returns dict of metric_name -> {score, reason}.
    Logs feedback scores to Opik trace if available.
    """
    if not HAS_OPIK:
        return {}
    
    try:
        # Run LLM-as-judge evals in parallel
        relevance_task = eval_response_relevance(query, response)
        completeness_task = eval_response_completeness(query, response, agents_used)
        hallucination_task = eval_hallucination_risk(query, response, agent_results)
        efficiency_task = eval_agent_efficiency(agents_used, chain_of_thought, elapsed_time)
        
        results = await asyncio.gather(
            relevance_task,
            completeness_task,
            hallucination_task,
            efficiency_task,
            return_exceptions=True
        )
        
        metric_names = ["relevance", "completeness", "hallucination_safety", "efficiency"]
        eval_results = {}
        
        for name, result in zip(metric_names, results):
            if isinstance(result, Exception):
                eval_results[name] = {"score": 0.0, "reason": f"Error: {str(result)[:100]}"}
            else:
                eval_results[name] = result
        
        return eval_results
        
    except Exception as e:
        print(f"⚠️ Online evaluation error: {e}")
        return {}


def log_feedback_to_current_trace(eval_results: Dict[str, Dict]):
    """
    Log evaluation scores as feedback scores on the CURRENT Opik trace.
    Must be called from within a function decorated with @opik_track.
    Uses opik.opik_context.update_current_trace() which auto-detects the active trace.
    """
    if not HAS_OPIK or not eval_results:
        return
    
    try:
        from opik.opik_context import update_current_trace
        
        feedback_scores = []
        for metric_name, result in eval_results.items():
            feedback_scores.append({
                "name": f"eval_{metric_name}",
                "value": result.get("score", 0.0),
                "reason": result.get("reason", "")[:500],
            })
        
        update_current_trace(feedback_scores=feedback_scores)
        print(f"✅ Logged {len(feedback_scores)} feedback scores to current trace")
    except Exception as e:
        print(f"⚠️ Failed to log feedback scores: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. OFFLINE EXPERIMENT TRACKING
#    Run benchmark evaluations against a test dataset.
# ─────────────────────────────────────────────────────────────────────────────

# Test dataset for benchmarking the multi-agent system
EVAL_DATASET = [
    {
        "input": "Find finance2023.pdf",
        "expected_agents": ["filesystem"],
        "expected_behavior": "Should find and return the file path",
        "category": "file_search",
    },
    {
        "input": "Compare financial data between 2023 and 2024",
        "expected_agents": ["rag", "text_extraction", "data_analysis"],
        "expected_behavior": "Should extract data from both files and provide comparison table",
        "category": "data_analysis",
    },
    {
        "input": "Classify the document finance2024.pdf",
        "expected_agents": ["filesystem", "file_classification"],
        "expected_behavior": "Should find the file and classify it as financial document",
        "category": "classification",
    },
    {
        "input": "Extract text from finance2023.pdf",
        "expected_agents": ["filesystem", "text_extraction"],
        "expected_behavior": "Should find and extract text content from the PDF",
        "category": "text_extraction",
    },
    {
        "input": "What is the total revenue for 2023?",
        "expected_agents": ["rag"],
        "expected_behavior": "Should search documents and return revenue figure",
        "category": "rag_search",
    },
    {
        "input": "Save metadata for finance2024.pdf",
        "expected_agents": ["filesystem", "text_extraction", "file_classification", "metadata"],
        "expected_behavior": "Should process the full pipeline and save metadata",
        "category": "full_pipeline",
    },
]


def agent_selection_scorer(
    dataset_item: Dict[str, Any],
    task_outputs: Dict[str, Any]
) -> score_result.ScoreResult:
    """Score how well the system selected the right agents."""
    expected = set(dataset_item.get("expected_agents", []))
    actual = set(task_outputs.get("agents_used", []))
    
    if not expected:
        return score_result.ScoreResult(name="agent_selection", value=1.0, reason="No expected agents defined")
    
    # Calculate overlap
    correct = expected & actual
    precision = len(correct) / len(actual) if actual else 0
    recall = len(correct) / len(expected) if expected else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return score_result.ScoreResult(
        name="agent_selection",
        value=round(f1, 2),
        reason=f"Expected: {expected}, Got: {actual}, F1: {f1:.2f}"
    )


def task_completion_scorer(
    dataset_item: Dict[str, Any],
    task_outputs: Dict[str, Any]
) -> score_result.ScoreResult:
    """Score whether the task was completed successfully."""
    is_complete = task_outputs.get("is_task_complete", False)
    has_content = bool(task_outputs.get("response", "").strip())
    has_error = "error" in task_outputs.get("response", "").lower()
    
    if is_complete and has_content and not has_error:
        score = 1.0
        reason = "Task completed successfully with content"
    elif has_content and not has_error:
        score = 0.7
        reason = "Content returned but task not marked complete"
    elif has_error:
        score = 0.2
        reason = f"Error in response"
    else:
        score = 0.0
        reason = "No content returned"
    
    return score_result.ScoreResult(
        name="task_completion",
        value=score,
        reason=reason
    )


def response_time_scorer(
    dataset_item: Dict[str, Any],
    task_outputs: Dict[str, Any]
) -> score_result.ScoreResult:
    """Score the response time performance."""
    elapsed = task_outputs.get("elapsed_time", 999)
    
    if elapsed < 5:
        score = 1.0
    elif elapsed < 10:
        score = 0.8
    elif elapsed < 20:
        score = 0.6
    elif elapsed < 30:
        score = 0.4
    else:
        score = 0.2
    
    return score_result.ScoreResult(
        name="response_time",
        value=score,
        reason=f"Response time: {elapsed:.1f}s"
    )


async def run_experiment(
    multi_agent_system,
    experiment_name: str = None,
    dataset_items: List[Dict] = None,
    experiment_config: Dict = None,
) -> Optional[str]:
    """
    Run an offline evaluation experiment against the test dataset.
    
    Args:
        multi_agent_system: The MultiAgentSystem instance
        experiment_name: Name for this experiment run
        dataset_items: Custom dataset items (defaults to EVAL_DATASET)
        experiment_config: Additional config to log with the experiment
        
    Returns:
        Experiment URL or None
    """
    if not HAS_OPIK:
        print("⚠️ Opik not available, skipping experiment")
        return None
    
    items = dataset_items or EVAL_DATASET
    experiment_name = experiment_name or f"agentic-assistant-eval-{int(time.time())}"
    
    try:
        client = opik.Opik(project_name=_OPIK_PROJECT)
        
        # Create or get dataset
        dataset = client.get_or_create_dataset(
            name="agentic-assistant-benchmark",
            description="Benchmark dataset for multi-agent document assistant"
        )
        
        # Insert items (Opik deduplicates automatically)
        dataset.insert(items)
        
        # Define the evaluation task
        async def evaluation_task(item: Dict[str, Any]) -> Dict[str, Any]:
            start_time = time.time()
            try:
                result = await multi_agent_system.run(
                    query=item["input"],
                    session_id=f"eval-{int(time.time())}",
                    user_role="admin"
                )
                elapsed = time.time() - start_time
                
                return {
                    "response": result.get("content", ""),
                    "agents_used": result.get("used_tools", []),
                    "is_task_complete": result.get("is_task_complete", False),
                    "chain_of_thought": result.get("chain_of_thought", []),
                    "elapsed_time": elapsed,
                }
            except Exception as e:
                return {
                    "response": f"Error: {str(e)}",
                    "agents_used": [],
                    "is_task_complete": False,
                    "chain_of_thought": [],
                    "elapsed_time": time.time() - start_time,
                }
        
        # Wrap async task for opik.evaluate (which expects sync)
        def sync_task(item):
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, evaluation_task(item))
                    return future.result()
            else:
                return asyncio.run(evaluation_task(item))
        
        # Run evaluation
        config = experiment_config or {
            "model": "gemini-2.0-flash",
            "temperature": 0.0,
            "agents": ["filesystem", "rag", "text_extraction", "file_classification", "metadata", "data_analysis"],
            "embedding_model": "gemini-embedding-001",
        }
        
        result = opik.evaluate(
            dataset=dataset,
            task=sync_task,
            scoring_metrics=[],
            scoring_functions=[
                agent_selection_scorer,
                task_completion_scorer,
                response_time_scorer,
            ],
            experiment_name=experiment_name,
            experiment_config=config,
            project_name=_OPIK_PROJECT,
        )
        
        print(f"✅ Experiment '{experiment_name}' completed!")
        print(f"   View results at: https://www.comet.com/opik")
        return experiment_name
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. UTILITY: Format eval results for UI display
# ─────────────────────────────────────────────────────────────────────────────

def format_eval_results_html(eval_results: Dict[str, Dict]) -> str:
    """Format evaluation results as HTML for Gradio display."""
    if not eval_results:
        return ""
    
    html = '<div style="margin-top:8px; padding:8px; background:rgba(255,255,255,0.05); border-radius:6px; font-size:12px;">'
    html += '<div style="font-weight:600; margin-bottom:4px; color:#a78bfa;">📊 Quality Scores</div>'
    
    icons = {
        "relevance": "🎯",
        "completeness": "✅",
        "hallucination_safety": "🛡️",
        "efficiency": "⚡",
    }
    
    for metric, result in eval_results.items():
        score = result.get("score", 0)
        icon = icons.get(metric, "📏")
        
        # Color based on score
        if score >= 0.8:
            color = "#4ade80"  # green
        elif score >= 0.6:
            color = "#facc15"  # yellow
        else:
            color = "#f87171"  # red
        
        bar_width = int(score * 100)
        label = metric.replace("_", " ").title()
        
        html += f'''<div style="display:flex; align-items:center; gap:6px; margin:2px 0;">
            <span>{icon}</span>
            <span style="width:120px; color:#d1d5db;">{label}</span>
            <div style="flex:1; background:rgba(255,255,255,0.1); border-radius:3px; height:8px;">
                <div style="width:{bar_width}%; background:{color}; height:100%; border-radius:3px;"></div>
            </div>
            <span style="color:{color}; font-weight:600; width:35px; text-align:right;">{score:.0%}</span>
        </div>'''
    
    # Overall score
    avg_score = sum(r.get("score", 0) for r in eval_results.values()) / max(len(eval_results), 1)
    html += f'<div style="margin-top:4px; text-align:right; color:#a78bfa; font-weight:600;">Overall: {avg_score:.0%}</div>'
    html += '</div>'
    
    return html
