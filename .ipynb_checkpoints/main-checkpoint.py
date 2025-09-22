
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimize MCTS optimal paths with TextGrad, leveraging sibling (non‑optimal) paths.

Usage example
-------------
python mcts_textgrad_optimizer.py \
  --input_jsonl  math_testset_annotation_100.jsonl \
  --output_jsonl optimized_paths.jsonl \
  --threads 8

Author: lovecambi (interfk@gmail.com)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from textgrad import Variable
import numpy as np
import textgrad as tg
from tqdm import tqdm
# ---------- utils: turn Node -> JSON path ----------
# ---------- math_is_equiv fallback ----------
try:
    from mcts_math.agents.utils import math_is_equiv  # 若项目里已有更智能实现
except ImportError:
    # 最简：去除空白直接比较
    def math_is_equiv(a: str, b: str) -> bool:
        return a.strip() == b.strip()
def leaf_to_steps(leaf: InferNode) -> str:
    """
    把根→该 leaf 的整条路径序列化为
        [{"step": ..., "Q": ...}, ...]
    - 只保留非空 text
    - 若 leaf.final_answer 与最后一步不同，则再补一条
    """
    chain = []
    n = leaf
    while n:                 # 反向回溯到根
        chain.append(n)
        n = n.parent
    chain.reverse()

    steps = [
        {"step": nd.text.strip(), "Q": nd.q_value}
        for nd in chain if nd.text and nd.text.strip()
    ]
    if leaf.final_answer and (not steps or steps[-1]["step"] != leaf.final_answer):
        steps.append({"step": leaf.final_answer, "Q": leaf.q_value})

    import json
    return json.dumps(steps, ensure_ascii=False)
def to_plain(obj):
    """
    Recursively turn tg.Variable → obj.value.
    Works for nested dict / list / tuple.
    """
    if isinstance(obj, Variable):
        return obj.value
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj
# ========== Ⅰ.  一些常量 ==========
NO_VALID_CHILD = "No valid child"
TOO_MANY_STEPS = "Too many steps"
TOO_MANY_CODE_ERRORS = "Too many code errors"
SAVE_INTERVAL = 10
LOCK = threading.Lock()

# ========== Ⅱ.  MCTS 树节点 ==========
class InferNode:
    """极简 Node；只保存本任务需要的字段"""
    def __init__(
        self,
        tag: str,
        text: str = "",
        final_answer: str = "",
        value: float = 0.0,
        q_value: float = 0.0,
        prior: float = 1.0,
        visit_count: int = 0,
        depth: int = 0,
        parent: Optional["InferNode"] = None,
    ):
        self.tag = tag
        self.text = text
        self.final_answer = final_answer
        self.value = value
        self.q_value = q_value
        self.prior = prior
        self.visit_count = visit_count
        self.depth = depth
        self.parent = parent
        self.children: List["InferNode"] = []

    # puct 用于排序可选
    def puct(self, c_puct: float = 1.25) -> float:
        q_val = self.q_value if self.visit_count > 0 else 0.0
        u_val = 0.0
        if self.parent:
            u_val = (
                c_puct
                * self.prior
                * np.sqrt(self.parent.visit_count)
                / (1 + self.visit_count)
            )
        return q_val + u_val


# ========== Ⅲ.  树重建 & 最优路径提取 ==========
def rebuild_tree(
    tree_dict: Dict[str, Any],
    c_puct: float = 1.25,
    root_tag: str = "0",
) -> Tuple[InferNode, int]:
    """
    • 兼容没给 '0' 节点的情况：自动创建一个空根，然后把所有 '0.xxx' 节点挂上去  
    • 不再假设孩子下标连续：直接根据 tag 前缀判定父子
    """
    DEFAULT_INFO = dict(text="", final_answer="", value=0.0,
                        q_value=0.0, prior=1.0, visit_count=0)

    # 若缺失根，补一条默认记录
    if root_tag not in tree_dict:
        tree_dict[root_tag] = DEFAULT_INFO.copy()

    # 先按 tag 深度升序创建所有节点
    nodes: Dict[str, InferNode] = {}
    for tag, info in sorted(tree_dict.items(), key=lambda kv: kv[0].count(".")):
        depth = tag.count(".")
        parent_tag = ".".join(tag.split(".")[:-1]) if tag != root_tag else None
        node = InferNode(
            tag=tag,
            text=info.get("text", ""),
            final_answer=info.get("final_answer", ""),
            value=info.get("value", 0.0),
            q_value=info.get("q_value", 0.0),
            prior=info.get("prior", 1.0),
            visit_count=info.get("visit_count", 0),
            depth=depth,
            parent=None,        # 先留空，稍后赋值
        )
        nodes[tag] = node

    # 再建立父子关系
    max_depth = 0
    for tag, node in nodes.items():
        if tag == root_tag:
            continue
        parent_tag = ".".join(tag.split(".")[:-1])
        parent = nodes.get(parent_tag)
        if parent is None:
            # 极端情况：父节点缺失 => 挂到根
            parent = nodes[root_tag]
        node.parent = parent
        parent.children.append(node)
        max_depth = max(max_depth, node.depth)

    return nodes[root_tag], max_depth


def is_valid_leaf(node: InferNode) -> bool:
    if node.children:
        return False
    if not node.final_answer:
        return False
    if node.final_answer in {NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS}:
        return False
    return True


def sort_nodes(nodes: List[InferNode], strategy: str = "q_value", c_puct: float = 1.25):
    key_map = {
        "q_value": lambda n: n.q_value,
        "value": lambda n: n.value,
        "visit_count": lambda n: n.visit_count,
        "puct": lambda n: n.puct(c_puct),
    }
    key_fn = key_map.get(strategy, key_map["q_value"])
    return sorted(nodes, key=key_fn, reverse=True)


def extract_best_and_siblings(
    tree_dict: Dict[str, Any],
    *,
    score_key: str = "q_value",         # 想改用 visit_count 就写 "visit_count"
    ground_truth: str | None = None,
    max_siblings: int = 3,
) -> Tuple[str, List[str]]:
    """
    返回:
        best_json          —— 最优叶完整路径 (JSON 字符串)
        sibling_jsons      —— 同父其它叶完整路径列表
    """
    root, _ = rebuild_tree(tree_dict)   # rebuild_tree 已改成“不需要 '0' 也能挂”

    # 1. 收集所有合法叶子
    leaves = []
    stack = [root]
    while stack:
        nd = stack.pop()
        if nd.children:
            stack.extend(nd.children)
        elif is_valid_leaf(nd):
            leaves.append(nd)

    if not leaves:
        return "", []

    # 2. 选最优叶：先看 ground_truth 是否匹配；否则按 score_key 最大
    best_leaf = None
    if ground_truth:
        for lf in leaves:
            if math_is_equiv(str(ground_truth), str(lf.final_answer)):
                best_leaf = lf
                break
    if best_leaf is None:
        best_leaf = max(leaves, key=lambda n: getattr(n, score_key))

    # 3. sibling = 同父下其它合法叶
    siblings = [
        lf for lf in (best_leaf.parent.children if best_leaf.parent else [])
        if lf is not best_leaf and is_valid_leaf(lf)
    ][:max_siblings]

    best_json = leaf_to_steps(best_leaf)
    sibling_jsons = collect_sibling_paths(
        best_leaf,
        score_key=score_key,
        max_per_level=2,   )

    best_json = leaf_to_steps(best_leaf)
    return best_json, sibling_jsons
def collect_sibling_paths(best_leaf, *, score_key="q_value", max_per_level=2):
    """
    遍历最优链的每一层:
        - 找到该层的所有兄弟
        - 对每个兄弟:
            · 若兄弟是叶 -> 直接序列化链
            · 否则 -> 沿 score_key 最大路径往下走到叶, 把整条链序列化
    返回: List[str] (每条都是 leaf_to_steps() 生成的 JSON 字符串)
    """
    sib_paths = []
    # 1. 先拿根→最优叶的所有节点
    chain = []
    n = best_leaf
    while n:
        chain.append(n)
        n = n.parent
    chain.reverse()         # 根在前

    for node in chain:      # node 是最优链里的一个节点
        if not node.parent:   # 根没有兄弟
            continue
        # 取 node.parent 的所有孩子, 过滤掉 node 自己
        for bro in node.parent.children:
            if bro is node:
                continue
            # a) 兄弟是叶
            if not bro.children:
                sib_paths.append(leaf_to_steps(bro))
            # b) 兄弟还有子树 -> 选子树里 score_key 最大的叶
            else:
                leaf = descend_to_best_leaf(bro, score_key)
                sib_paths.append(leaf_to_steps(leaf))
            if len(sib_paths) >= max_per_level * len(chain):
                break
    return sib_paths


def descend_to_best_leaf(sub_root, score_key):
    """DFS/BFS 找到 sub_root 子树里 score_key 最大的合法叶"""
    best = None
    stack = [sub_root]
    while stack:
        nd = stack.pop()
        if nd.children:
            stack.extend(nd.children)
        elif is_valid_leaf(nd):
            if (best is None) or getattr(nd, score_key) > getattr(best, score_key):
                best = nd
    return best or sub_root   # 极端情况

# ========== Ⅳ.  构造 TextGrad 数据 ==========
def make_steps_from_answer(answer_str: str, q_value: float = 0.0):
    """
    把纯文本 answer 封装为可与旧代码兼容的 JSON list 格式
    """
    return json.dumps([{"step": answer_str.strip(), "Q": q_value}], ensure_ascii=False)


def preprocess_sample(item: Dict[str, Any]) -> Dict[str, Any]:
    # 1. 提取最优路径 & sibling 路径（都是 JSON‑字符串）
    best_json, sibling_jsons = extract_best_and_siblings(
        item["react"],
        ground_truth=item.get("answer"),
        score_key="q_value",      # or "visit_count"
        max_siblings=3,
    )

    # 若极端情况下没取到合法叶，简单兜底为空串
    if not best_json:
        best_json = "[]"
        print("empty")

    return {
        "problem": tg.Variable(
            value=item["question"],
            role_description="math problem",
            requires_grad=False,
        ),
        "ground_truth": tg.Variable(
            value=item.get("answer", ""),
            role_description="ground truth",
            requires_grad=False,
        ),
        # ------ 未优化最佳路径 ------
        "original_output": tg.Variable(
            value=best_json,
            role_description="step‑by‑step best path",
            requires_grad=True,
        ),
        # ------ 同级非最优完整路径 ------
        "sibling_outputs": sibling_jsons,   # list[str]
    }


# ========== Ⅴ.  帮助：记录 Prompt / Response ==========
def add_call_log(call_log: List[Dict[str, str]], step: str, prompt: str, response: str):
    call_log.append({"step": step, "prompt": prompt, "response": response})


# ---------- 替换 LoggedFormattedCall -------------

class LoggedFormattedCall(tg.autograd.FormattedLLMCall):
    """
    扩展以记录 prompt / response。
    textgrad 原类并未公开 render_prompt()，因此手动 format。
    """
    def __init__(self, *args, log_container: list, step_name: str = "unknown", **kwargs):
        # 把 format_string / fields 取出来以便后续拼 prompt
        self._format_string: str = kwargs.get("format_string")  # 已由父类接收
        self._fields_dict: dict = kwargs.get("fields")          # "
        self._sys_prompt_var = kwargs.get("system_prompt")      # 可能是 Variable
        self._log_container = log_container
        self._step_name = step_name
        super().__init__(*args, **kwargs)

    def _render_prompt_text(self, inputs) -> str:
        """
        把 Variable -> value，补齐 inputs（如果有动态字段），返回最终字符串
        """
        merged = dict(self._fields_dict or {})
        if inputs:
            merged.update(inputs)
        def _val(x):
            return x.value if isinstance(x, tg.Variable) else x
        plain = {k: _val(v) for k, v in merged.items()}
        user_part = self._format_string.format(**plain)
        sys_part = _val(self._sys_prompt_var) if self._sys_prompt_var else ""
        return (sys_part + "\n" + user_part).strip()
    def forward(self, inputs: dict[str, tg.Variable], response_role_description=None) -> tg.Variable:
    # 先格式化 prompt 文本
        merged = dict(self._fields_dict or {})
        if inputs:
            merged.update(inputs)
        def _val(x):
            return x.value if isinstance(x, tg.Variable) else x
        plain = {k: _val(v) for k, v in merged.items()}
        formatted_input_string = self._format_string.format(**plain)
        system_prompt_value = _val(self._sys_prompt_var) if self._sys_prompt_var else None
    
        # 调用 engine，**这里改为直接拿完整响应**
        response = self.engine(
            formatted_input_string,
            system_prompt=system_prompt_value,
            return_full_response=True  # 假设你能传这个参数让 engine 返回完整 response
        )
    
        # response 是 dict，取文本
        response_text = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
    
        # 返回带有 extra_info 的 Variable，方便 __call__ 访问
        out_var = tg.Variable(
            value=response_text,
            predecessors=[self._sys_prompt_var] + list(inputs.values()) if self._sys_prompt_var else list(inputs.values()),
            role_description=response_role_description or tg.VARIABLE_OUTPUT_DEFAULT_ROLE
        )
        # 自定义加一个字段保存完整响应（方便外部访问 usage）
        out_var.extra_info = response
    
        # 设置梯度信息等（保持不变）
        out_var.set_grad_fn(
            tg.BackwardContext(
                backward_fn=self.backward,
                response=out_var,
                prompt=formatted_input_string,
                system_prompt=system_prompt_value,
            )
        )
    
        return out_var

    def __call__(self, inputs=None, response_role_description=None):
        prompt_txt = self._render_prompt_text(inputs)
        # === 真正调用 LLM ===
        out_var = super().__call__(inputs=inputs, response_role_description=response_role_description)
        if out_var.extra_info is not None:
        if hasattr(out_var.extra_info, "usage"):
            usage = out_var.extra_info.usage
        elif hasattr(out_var.extra_info, "to_dict"):
            usage = out_var.extra_info.to_dict().get("usage", {})
        else:
            usage = {}
    else:
        usage = {}
        # === 记录 ===
        self._log_container.append({
            "step": self._step_name,
            "prompt": prompt_txt,
            "response": out_var.value,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        })
        return out_var

# ========== Ⅵ.  DataSet Optimizer ==========
class DatasetOptimizer:
    def __init__(self, input_path: str, output_path: str, resume=True, threads=8):
        self.input_path = input_path
        self.output_path = output_path
        self.partial_path = output_path.replace(".jsonl", "_partial.jsonl")
        self.threads = threads

        self.engine = tg.get_engine("Qwen2.5-Math-72B")          # 按需换成 openai:gpt-4 等
        tg.set_backward_engine(self.engine)

        self.samples_raw: List[Dict[str, Any]] = []
        self.processed_samples: List[Dict[str, Any]] = []
        self.optimized_results: Dict[str, Dict[str, Any]] = {}

        # 加载
        self._load_dataset()
        self._load_partial(resume)

    def _load_dataset(self):
        with open(self.input_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples_raw.append(json.loads(line))
        # 预处理
        for item in tqdm(self.samples_raw, desc="Preprocess"):
            self.processed_samples.append(preprocess_sample(item))

    def _load_partial(self, resume=True):
        if resume and os.path.exists(self.partial_path):
            with open(self.partial_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    self.optimized_results[obj["idx"]] = obj
            print(f"[Resume] loaded {len(self.optimized_results)} lines from partial file.")
# ========= util: recursively cast =========
    def to_plain(obj):
        """Recursively turn tg.Variable → obj.value  (dict / list / tuple supported)"""
        from textgrad import Variable
        if isinstance(obj, Variable):
            return obj.value
        if isinstance(obj, dict):
            return {k: to_plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_plain(v) for v in obj]
        return obj
    def _save_partial(self):
        with LOCK, open(self.partial_path, "w", encoding="utf-8") as f:
            for obj in self.optimized_results.values():
                f.write(json.dumps(to_plain(obj), ensure_ascii=False) + "\n")
        print(f"[Auto‑save] {len(self.optimized_results)} results written.")
    # -------- TextGrad loss & optimisation --------
    def create_critique_call(self, sample, log_container):
        """
        Critique prompt: 让 LLM 找问题 + 给改进建议
        """
        sys_prompt = tg.Variable(
            value=
            "You are a brilliant math‑solution reviewer.Your task is to identify both the positive and negative aspects of the sibling answer in order to further improve the Original answer.\n"
            "You do NOT solve from scratch; you analyse the given answer, "
            "compare with sibling answers, find flaws, and suggest concrete improvements."
            "IMPORTANT:\n"
            "1. All output must be a hierarchical, numbered outline that mirrors the original answer’s tree structure (e.g. 0.1, 0.1.1, 0.1.1.1, etc.).\n"
            "2. For **each depth level**, provide **exactly one** concise improvement suggestion synthesizing all nodes at that level:\n"
            "   • Level 1 (0.x): one suggestion covering 0.1, 0.2, 0.3…\n"
            "   • Level 2 (0.x.x): one suggestion covering 0.1.1, 0.1.2, 0.2.1…\n"
            "   • Level 3 (0.x.x.x): one suggestion covering 0.2.2.1, 0.2.2.2…\n",

            requires_grad=False,
            role_description="system prompt for critique",
        )

        fmt = (
            "Question:\n{problem}\n\n"
            "Original answer (to be improved):\n{orig}\n\n"
            "Sibling answers from the same search layer (for reference, may contain good clues or mistakes,You should identify some strengths to improve the original answer, and also find some negative aspects (weaknesses) to make the original answer better.):\n"
            "{siblings}\n\n"
            "Ground‑truth short answer (do NOT just copy, use it to check correctness!):\n{gt}\n\n"
            "Tasks:\n"
            "1. Point out any logical errors, missing steps, or incorrect conclusions in the original answer(including its code).\n"
            "2. Identify useful details present in sibling answers but missing in the original.\n"
            "3. Summarise clear bullet‑point improvements.\n"
            "Give the bullet points only, no extra chit‑chat."
        )
        import json
        sib_preview = "\n\n".join(
            f"- {json.loads(js)[-1]['step']}"
            for js in sample["sibling_outputs"]
        ) or "None"
        fields = {
            "problem": sample["problem"],
            "orig": sample["original_output"],
            "siblings": tg.Variable(value=sib_preview,
                                    role_description="sibling answers",
                                    requires_grad=False),
            "gt": sample["ground_truth"],
        }

        # --
        return LoggedFormattedCall(
            engine=self.engine,
            format_string=fmt,
            fields=fields,
            system_prompt=sys_prompt,
            log_container=log_container,
            step_name="critique",
        )

    def create_revision_call(self, sample, critique_feedback_var, log_container):
        """
        Revision prompt: 按反馈重写答案
        """
        sys_prompt = tg.Variable(
            value="You are a meticulous mathematician. Rewrite the answer cleanly.",
            requires_grad=False,
            role_description="system prompt for revision",
        )
        fmt = (
            "Question:\n{problem}\n\n"
            "Original answer:\n{orig}\n\n"
            "Ground-truth short answer:\n{gt}\n\n"
            "Critique feedback (bullet points):\n{fb}\n\n"
            "Rewrite the answer so that:\n"
            "- All listed issues are fixed\n"
            "- Useful insights are incorporated\n"
            "- The final numeric/closed-form answer matches ground-truth\n"
            "Return ONLY the rewritten full answer, nothing else.\n"
            "Format Rules:\n"
            "1. Number each reasoning step clearly.  \n"
            "- Begin each paragraph with “Step 1:”, “Step 2:”, “Step 3:”, etc.  \n"
            "- Each step must cover exactly one idea, operation, or stage in your reasoning, written entirely in prose.\n"
            "2.  Absolutely no code. \n"
            "- Do **not** include any code, pseudocode, or code-like snippets anywhere. \n"
            "- Describe any procedures or algorithms fully in words.\n"
            "3. State the final result on its own line, using this exact phrasing: \n"
            "- The answer is: $18$ Replace “18” with your computed result inside the dollar signs.\n"
            "4. Be as detailed as possible. \n"
            "- Your explanation must be at least as long as the original and preferably longer. \n"
            "- Unpack every intermediate thought, justification, and clarification—do not skip any logical connections.\n"
            "5. Structure your output in this order: \n"
            "- Numbered reasoning steps (Step 1, Step 2, …) \n"
            "- A single final line: \n"
            "The answer is: $<your answer here>$\n"
            "Thank you—obey these rules strictly and deliver a richly detailed, step-by-step explanation with absolutely no code.\n"
        )
        fields = {
            "problem": sample["problem"],
            "orig": sample["original_output"],
            "gt": sample["ground_truth"],      # ← 直接用
            "fb": critique_feedback_var,
        }
        return LoggedFormattedCall(
            engine=self.engine,
            format_string=fmt,
            fields=fields,
            system_prompt=sys_prompt,
            log_container=log_container,
            step_name="revision",
        )

    # -------- 单条样本优化 --------
    def _optimize_one(self, idx: int, sample_processed: Dict[str, Any]):
        """
        • 不重复优化已完成样本  
        • 整个优化过程的 prompt/response 记录在 call_log
        """
        if str(idx) in self.optimized_results:
            return
        call_log: List[Dict[str, str]] = []

        # === 1. Critique ===
        critique_call = self.create_critique_call(sample_processed, call_log)
        import json
        sib_preview = "\n\n".join(
            f"- {json.loads(js)[-1]['step']}" for js in sample_processed["sibling_outputs"]
        ) or "None"

        inputs_dict = {
            "problem": sample_processed["problem"],
            "orig": sample_processed["original_output"],
            "siblings": tg.Variable(value=sib_preview,
                                    role_description="sibling answers",
                                    requires_grad=False),
            "gt": sample_processed["ground_truth"],
        }

        critique_feedback_var = critique_call(
                inputs=inputs_dict,                       # ← 关键：传入 dict
                response_role_description="critique feedback",
        )

            #
            # ---------- 2. Revision ----------
            #
        revision_call = self.create_revision_call(sample_processed, critique_feedback_var, call_log)

        revision_inputs = {
                "problem": sample_processed["problem"],
                "orig": sample_processed["original_output"],
                "gt": sample_processed["ground_truth"],
                "fb": critique_feedback_var,
        }

        revised_answer_var = revision_call(
                inputs=revision_inputs,                  # ← 同理
                response_role_description="rewritten answer",
        )

    
        # === 3. (可选) TextGrad backward ===
        # 这里如果想用 textgrad 的 Variable 梯度特性继续 fine‑tune，可保留
        # 简单起见：一次性重写即可
        # -----
        with LOCK:
            self.optimized_results[str(idx)] = {
                "idx": str(idx),
                "question": sample_processed["problem"].value,
                "answer": sample_processed["ground_truth"],
                "original_output": sample_processed["original_output"].value,
                "optimized_output": revised_answer_var.value,
                "call_log": call_log,
            }

    # -------- 入口：并行优化整个数据集 --------
    def optimise_all(self):
        total = len(self.processed_samples)
        futures = []
        with ThreadPoolExecutor(max_workers=self.threads) as ex, tqdm(total=total, desc="Optimising") as pbar:
            for idx, sample_p in enumerate(self.processed_samples):
                if str(idx) in self.optimized_results:
                    pbar.update(1)
                    continue
                fut = ex.submit(self._optimize_one, idx, sample_p)
                fut.add_done_callback(lambda _: pbar.update(1))
                futures.append(fut)

                if len(futures) % SAVE_INTERVAL == 0:
                    for f in as_completed(futures):
                        f.result()
                    futures = []
                    self._save_partial()

            # wait remaining
            for f in as_completed(futures):
                f.result()
            self._save_partial()

    # -------- 保存最终输出 --------
    def save_final(self):
        with open(self.output_path, "w", encoding="utf-8") as f:
            for idx in range(len(self.samples_raw)):
                if str(idx) in self.optimized_results:
                    f.write(json.dumps(to_plain(self.optimized_results[str(idx)]),
                                    ensure_ascii=False) + "\n")
        print(f"[Done] {len(self.optimized_results)} lines saved to {self.output_path}")

# ========== Ⅶ.  命令行 ==========
def parse_args():
    parser = argparse.ArgumentParser(description="MCTS‑>TextGrad  最优路径优化器")
    parser.add_argument("--input_jsonl", default='/root/autodl-tmp/RSTAR/qwenmath/Qwen2.5Math_rstar_pruned_all.jsonl', help="原始包含 react 树的 JSONL 文件")
    parser.add_argument("--output_jsonl", default="/root/autodl-tmp/RSTAR/qwenmath/Qwen2.5Math_rstar_pruned_all_done.jsonl", help="输出 JSONL")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--no_resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    opt = DatasetOptimizer(
        input_path=args.input_jsonl,
        output_path=args.output_jsonl,
        resume=not args.no_resume,
        threads=args.threads,
    )
    opt.optimise_all()
    opt.save_final()


if __name__ == "__main__":
    # windows utf‑8
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    main()
