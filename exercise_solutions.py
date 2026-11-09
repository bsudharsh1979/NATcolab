"""
exercise_solutions.py — programmatic solutions for the NATcolab exercises.

The DLI lab ships pre-edited files under /dli/task/Exercise_Solutions/ and asks you to `cp` them over the
working copies. In Colab the same edits are applied by these idempotent helpers, so that
"Runtime → Run all" works end-to-end. Try the exercise by hand first (edit the files in Colab's file
browser), then call the helper to compare or to catch up. `NATClient.reset()` undoes everything.

    import exercise_solutions as sol
    sol.exercise_1_temperature(0.1)
    sol.exercise_2_plot_color()
    sol.exercise_3_action_recommendation()
    sol.enable_phoenix_tracing()
    sol.optional_codegen()
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

ALM_HOME = Path(os.environ.get("ALM_HOME", Path(__file__).resolve().parent))
CONFIG = ALM_HOME / "configs" / "config-reasoning.yml"
SRC = ALM_HOME / "asset_lifecycle_management_agent" / "src" / "asset_lifecycle_management_agent"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


def _ok(msg: str) -> None:
    print(f"✅ {msg}")


def _validate_config() -> dict:
    cfg = yaml.safe_load(_read(CONFIG))
    assert isinstance(cfg, dict) and "functions" in cfg and "llms" in cfg, "config-reasoning.yml is not valid"
    return cfg


def _add_function_block(text: str, name: str, block_lines: list[str], after: str = "plot_anomaly") -> str:
    """Insert a `functions:` entry (2-space indented) right after the `after:` entry."""
    if re.search(rf"^  {name}:\s*$", text, flags=re.M):
        return text
    m = re.search(rf"^  {after}:\s*\n(?:    .*\n)+", text, flags=re.M)
    if not m:
        raise RuntimeError(f"Could not find the '{after}' function block in the config")
    block = "\n  " + name + ":\n" + "".join(f"    {line}\n" for line in block_lines)
    return text[:m.end()] + block + text[m.end():]


def _add_tool_name(text: str, tool: str) -> str:
    m = re.search(r"(tool_names:\s*\[)(.*?)(\])", text, flags=re.S)
    if not m:
        raise RuntimeError("Could not find data_analysis_assistant.tool_names in the config")
    body = m.group(2)
    if re.search(rf'"{tool}"|\b{tool}\b', body):
        return text
    items = [i for i in body.rstrip().splitlines() if i.strip()]
    indent = re.match(r"\s*", items[-1]).group(0) if items else "      "
    new_body = body.rstrip().rstrip(",") + f',\n{indent}"{tool}"\n    '
    return text[:m.start(2)] + new_body + text[m.end(2):]


def _add_prompt_line(text: str, anchor_regex: str, new_line: str) -> str:
    if new_line.strip() in text:
        return text
    m = re.search(anchor_regex, text, flags=re.M)
    if not m:
        raise RuntimeError(f"Could not find prompt anchor {anchor_regex!r}")
    indent = re.match(r"[ \t]*", m.group(0)).group(0)
    return text[:m.end()] + "\n" + indent + new_line.strip() + text[m.end():]


# ------------------------------------------------------------------------------- Exercise 2.1
def exercise_1_temperature(temperature: float = 0.1) -> None:
    """Exercise 2.1 — set `temperature` on reasoning_llm."""
    text = _read(CONFIG)
    m = re.search(r"^  reasoning_llm:\s*\n((?:    .*\n)+)", text, flags=re.M)
    if not m:
        raise RuntimeError("reasoning_llm block not found")
    body = m.group(1)
    if re.search(r"^    temperature:.*$", body, flags=re.M):
        body2 = re.sub(r"^    temperature:.*$", f"    temperature: {temperature}  # Exercise 2.1", body, flags=re.M)
    else:
        body2 = body + f"    temperature: {temperature}  # Exercise 2.1\n"
    _write(CONFIG, text[:m.start(1)] + body2 + text[m.end(1):])
    assert _validate_config()["llms"]["reasoning_llm"]["temperature"] == temperature
    _ok(f"reasoning_llm.temperature = {temperature}")


# ------------------------------------------------------------------------------- Exercise 2.2
def exercise_2_plot_color() -> None:
    """Exercise 2.2 — add a `color` parameter to plot_line_chart (tool, plot utils and prompt)."""
    tool = SRC / "plotting" / "plot_line_chart_tool.py"
    s = _read(tool)
    if "color: str = Field(" not in s:
        s = s.replace(
            '        plot_title: str = Field(description="The title for the plot", default="Line Chart")\n',
            '        plot_title: str = Field(description="The title for the plot", default="Line Chart")\n'
            '        color: str = Field(description="The color for the line chart", default="blue")\n')
        s = s.replace(
            "async def _response_fn(data_json_path: str, x_axis_column: str, y_axis_column: str, plot_title: str) -> str:",
            "async def _response_fn(data_json_path: str, x_axis_column: str, y_axis_column: str, plot_title: str,\n"
            "                           color: str = \"blue\") -> str:")
        s = s.replace("                title=plot_title\n            )",
                      "                title=plot_title,\n                color=color\n            )")
        s = s.replace("    - plot_title: Title for the plot\n    \n    Output:",
                      "    - plot_title: Title for the plot\n    - color: Line color (e.g. 'red', 'green', '#76b900'); default blue\n    \n    Output:")
        _write(tool, s)
    assert "color=color" in _read(tool), "plot_line_chart_tool.py patch failed (upstream changed?)"

    utils = SRC / "plotting" / "plot_utils.py"
    u = _read(utils)
    if "color: str = \"blue\") -> Tuple[str, Optional[str]]" not in u:
        u = u.replace(
            "def create_line_chart(output_dir: str, data_json_path: str, x_col: str, \n                     y_col: str, title: str) -> Tuple[str, Optional[str]]:",
            "def create_line_chart(output_dir: str, data_json_path: str, x_col: str, \n                     y_col: str, title: str,\n                     color: str = \"blue\") -> Tuple[str, Optional[str]]:")
        # only the line-chart trace: it is the one followed by the line_chart hovertemplate
        start = u.index("def create_line_chart(")
        end = u.index("def create_distribution_plot(")
        seg = u[start:end].replace("line=dict(color='#1f77b4', width=3),", "line=dict(color=color, width=3),") \
                          .replace("marker=dict(size=6, color='#1f77b4'),", "marker=dict(size=6, color=color),")
        u = u[:start] + seg + u[end:]
        _write(utils, u)
    assert "line=dict(color=color, width=3)" in _read(utils), "plot_utils.py patch failed (upstream changed?)"

    text = _read(CONFIG)
    text = _add_prompt_line(
        text, r"^[ \t]*- plot_line_chart: to plot line charts between two columns of a dataset\.$",
        "- plot_line_chart: Extract color preference from user prompt (e.g., \"plot in blue\", \"use red color\") "
        "and pass it as the color parameter.")
    _write(CONFIG, text)
    _validate_config()
    _ok("plot_line_chart now accepts a `color` argument (tool + plot_utils + system prompt)")


# ------------------------------------------------------------------------------- Exercise 2.3
ACTION_TOOL_SOURCE = '''# SPDX-License-Identifier: Apache-2.0
"""Action recommendation tool (NATcolab solution for Exercise 2.3)."""

import logging

from pydantic import BaseModel, Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class ActionRecommendationToolConfig(FunctionBaseConfig, name="action_recommendation_tool"):
    """Generate actionable maintenance recommendations from analysis results."""
    llm_name: LLMRef = Field(description="The LLM used to generate the recommendations")


@register_function(config_type=ActionRecommendationToolConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def action_recommendation_tool(config: ActionRecommendationToolConfig, builder: Builder):

    class ActionRecommendationInputSchema(BaseModel):
        analysis_result: str = Field(description="The analysis output (e.g. RUL prediction, anomaly summary) "
                                                 "to base the recommendations on")
        context: str = Field(description="Additional context such as engine unit, dataset and sensor", default="")

    from langchain_core.prompts import ChatPromptTemplate

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior reliability engineer for turbofan engines. Given an analysis result, produce 3-5 "
         "specific, actionable maintenance recommendations. Format them as a numbered list. Start each item with "
         "an urgency level in square brackets: [URGENT], [HIGH], [MEDIUM] or [LOW], followed by the action and a "
         "short justification that references the numbers in the analysis. Plain text only."),
        ("user", "Analysis result:\\n{analysis_result}\\n\\nContext:\\n{context}"),
    ])
    chain = prompt | llm

    async def _response_fn(analysis_result: str, context: str = "") -> str:
        try:
            response = await chain.ainvoke({"analysis_result": analysis_result, "context": context or "n/a"})
            text = response.content if hasattr(response, "content") else str(response)
            return "MAINTENANCE RECOMMENDATIONS\\n\\n" + text.strip()
        except Exception as e:  # pragma: no cover - network errors
            logger.error("action_recommendation_tool failed: %s", e)
            return f"Error generating recommendations: {e}"

    yield FunctionInfo.from_fn(
        _response_fn,
        input_schema=ActionRecommendationInputSchema,
        description=("Generate actionable maintenance recommendations (with URGENT/HIGH/MEDIUM/LOW urgency) "
                     "from analysis results such as predicted or ground-truth RUL, anomaly detection output or "
                     "sensor trends. Input: analysis_result (text), context (engine unit, dataset, ...)."))
'''


def exercise_3_action_recommendation(llm_name: str = "reasoning_llm") -> None:
    """Exercise 2.3 — create, register and wire the action_recommendation tool."""
    rec_dir = SRC / "recommenders"
    rec_dir.mkdir(exist_ok=True)
    (rec_dir / "__init__.py").touch()
    _write(rec_dir / "action_recommendation_tool.py", ACTION_TOOL_SOURCE)

    reg = SRC / "register.py"
    r = _read(reg)
    line = "from .recommenders import action_recommendation_tool"
    if line not in r:
        _write(reg, r.rstrip("\n") + f"\n{line}\n")

    text = _read(CONFIG)
    text = _add_function_block(text, "action_recommendation",
                               ["_type: action_recommendation_tool", f'llm_name: "{llm_name}"'])
    text = _add_tool_name(text, "action_recommendation")
    text = _add_prompt_line(
        text, r"^[ \t]*- Use plot_anomaly to create interactive visualizations of anomaly detection results\.$",
        "- Use action_recommendation to provide actionable maintenance recommendations based on any analysis "
        "results (call it after completing the analysis to give users specific next steps).")
    _write(CONFIG, text)
    cfg = _validate_config()
    assert "action_recommendation" in cfg["functions"]["data_analysis_assistant"]["tool_names"]
    _ok("action_recommendation tool created (recommenders/), registered (register.py) and wired (config)")


# ------------------------------------------------------------------------------- Exercise 3.1
def enable_phoenix_tracing(endpoint: str = "http://localhost:6006/v1/traces",
                           project: str = "asset-lifecycle-management") -> None:
    """Exercise 3.1 — uncomment / add the Phoenix tracing block under general.telemetry."""
    text = _read(CONFIG)
    cfg = yaml.safe_load(text)
    tracing = ((cfg.get("general") or {}).get("telemetry") or {}).get("tracing") or {}
    if any((v or {}).get("_type") == "phoenix" for v in tracing.values()):
        _ok("Phoenix tracing already enabled")
        return
    pattern = (r"^([ \t]*)#[ \t]*tracing:[ \t]*\n[ \t]*#[ \t]*phoenix:[ \t]*\n[ \t]*#[ \t]*_type:[ \t]*phoenix[ \t]*\n"
               r"[ \t]*#[ \t]*endpoint:.*\n[ \t]*#[ \t]*project:.*\n")
    m = re.search(pattern, text, flags=re.M)
    ind = m.group(1) if m else "    "
    block = (f"{ind}tracing:\n{ind}  phoenix:\n{ind}    _type: phoenix\n"
             f"{ind}    endpoint: {endpoint}\n{ind}    project: {project}\n")
    if m:
        text = text[:m.start()] + block + text[m.end():]
    else:
        m2 = re.search(r"^  telemetry:\s*\n", text, flags=re.M)
        if not m2:
            raise RuntimeError("general.telemetry section not found")
        text = text[:m2.end()] + block + text[m2.end():]
    _write(CONFIG, text)
    assert _validate_config()["general"]["telemetry"]["tracing"]["phoenix"]["_type"] == "phoenix"
    _ok(f"Phoenix tracing enabled → {endpoint} (project '{project}')")


# ------------------------------------------------------------------------------- Optional notebook
def optional_codegen(coding_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b",
                     sandbox_uri: str = "http://127.0.0.1:6000/execute") -> None:
    """Optional notebook — add the coding LLM, code_generation_assistant + code_execution and wire them."""
    text = _read(CONFIG)
    if not re.search(r"^  coding_llm:\s*$", text, flags=re.M):
        m = re.search(r"^  multimodal_judging_llm:\s*\n(?:    .*\n)+", text, flags=re.M)
        block = ("\n  # Python code generation model (Optional notebook)\n  coding_llm:\n    _type: nim\n"
                 f'    model_name: "{coding_model}"\n    max_tokens: 16384\n')
        text = text[:m.end()] + block + text[m.end():]
    text = _add_function_block(text, "code_execution",
                               ["_type: code_execution", f"uri: {sandbox_uri}", 'sandbox_type: "local"',
                                "timeout: 60", "max_output_characters: 2000"])
    text = _add_function_block(text, "code_generation_assistant",
                               ["_type: code_generation_assistant", 'llm_name: "coding_llm"',
                                'code_execution_tool: "code_execution"', 'output_folder: "agent_workspace"',
                                "verbose: true", "max_retries: 1"], after="code_execution")
    text = _add_tool_name(text, "code_generation_assistant")
    text = _add_prompt_line(
        text, r"^[ \t]*- Tools save their outputs in the agent_workspace folder\..*$",
        "- When giving instructions to the code_generation_assistant, use only the filename itself (for example, "
        "filename.json). Do not include any folder paths, since the code_generation_assistant already operates "
        "within the outputs directory. Provide complete instructions (data files, columns, outputs) in a single "
        "parameter and use it ONLY for custom processing that the other tools cannot do.")
    _write(CONFIG, text)
    cfg = _validate_config()
    assert "code_generation_assistant" in cfg["functions"]["data_analysis_assistant"]["tool_names"]
    _ok(f"code generation enabled (coding_llm={coding_model}, sandbox={sandbox_uri})")
