# try:
#     from openai import OpenAI, AzureOpenAI
# except ImportError:
#     raise ImportError(
#         "If you'd like to use OpenAI models, please install the openai package by running `pip install openai`, "
#         "and add 'OPENAI_API_KEY' to your environment variables."
#     )

# import os
# import json
# import base64
# import platformdirs
# from tenacity import (
#     retry,
#     stop_after_attempt,
#     wait_random_exponential,
# )
# from typing import List, Union

# from .base import EngineLM, CachedEngine
# from .engine_utils import get_image_type_from_bytes

# # Default base URL for OLLAMA (not used by default)
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# class ChatOpenAI(EngineLM, CachedEngine):
#     DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

#     def __init__(
#         self,
#         model_string: str = "gpt-3.5-turbo-0613",
#         system_prompt: str = DEFAULT_SYSTEM_PROMPT,
#         is_multimodal: bool = False,
#         base_url: str = None,
#         **kwargs
#     ):
#         """
#         :param model_string: model name
#         :param system_prompt: system prompt text
#         :param is_multimodal: support multimodal input
#         :param base_url: custom API base URL (overrides OPENAI_API_BASE env)
#         """
#         # setup cache
#         root = platformdirs.user_cache_dir("textgrad")
#         cache_path = os.path.join(root, f"cache_openai_{model_string}.db")
#         super().__init__(cache_path=cache_path)

#         self.system_prompt = system_prompt
#         self.base_url = base_url or os.getenv("OPENAI_API_BASE")
#         self.model_string = model_string
#         self.is_multimodal = is_multimodal

#         api_key = os.getenv("OPENAI_API_KEY")
#         if not api_key:
#             raise ValueError(
#                 "Please set the OPENAI_API_KEY environment variable to use OpenAI models."
#             )

#         # initialize OpenAI client with optional custom base_url
#         client_kwargs = {"api_key": api_key}
#         if self.base_url:
#             client_kwargs["base_url"] = self.base_url

#         self.client = OpenAI(**client_kwargs)

#     @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
#     def generate(
#         self,
#         content: Union[str, List[Union[str, bytes]]],
#         system_prompt: str = None,
#         **kwargs
#     ):
#         if isinstance(content, str):
#             return self._generate_from_single_prompt(content, system_prompt=system_prompt, **kwargs)
#         else:
#             return self._generate_from_multiple_input(content, system_prompt=system_prompt, **kwargs)

#     def _generate_from_single_prompt(
#         self,
#         prompt: str,
#         system_prompt: str = None,
#         temperature: float = 0,
#         max_tokens: int = 2000,
#         top_p: float = 0.99,
#     ):
#         sys_prompt_arg = system_prompt or self.system_prompt

#         cache_or_none = self._check_cache(sys_prompt_arg + prompt)
#         if cache_or_none is not None:
#             return cache_or_none

#         response = self.client.chat.completions.create(
#             model=self.model_string,
#             messages=[
#                 {"role": "system", "content": sys_prompt_arg},
#                 {"role": "user", "content": prompt},
#             ],
#             temperature=temperature,
#             max_tokens=max_tokens,
#             top_p=top_p,
#             frequency_penalty=0,
#             presence_penalty=0,
#         )

#         text = response.choices[0].message.content
#         self._save_cache(sys_prompt_arg + prompt, text)
#         return text

#     def __call__(self, prompt: Union[str, List[Union[str, bytes]]], **kwargs):
#         return self.generate(prompt, **kwargs)

#     def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
#         formatted = []
#         for item in content:
#             if isinstance(item, bytes):
#                 img_type = get_image_type_from_bytes(item)
#                 b64 = base64.b64encode(item).decode("utf-8")
#                 formatted.append({
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/{img_type};base64,{b64}"},
#                 })
#             else:
#                 formatted.append({"type": "text", "text": item})
#         return formatted

#     def _generate_from_multiple_input(
#         self,
#         content: List[Union[str, bytes]],
#         system_prompt: str = None,
#         temperature: float = 0,
#         max_tokens: int = 2000,
#         top_p: float = 0.99,
#     ):
#         sys_prompt_arg = system_prompt or self.system_prompt
#         formatted = self._format_content(content)

#         cache_key = sys_prompt_arg + json.dumps(formatted)
#         cache_or_none = self._check_cache(cache_key)
#         if cache_or_none is not None:
#             return cache_or_none

#         response = self.client.chat.completions.create(
#             model=self.model_string,
#             messages=[
#                 {"role": "system", "content": sys_prompt_arg},
#                 {"role": "user", "content": formatted},
#             ],
#             temperature=temperature,
#             max_tokens=max_tokens,
#             top_p=top_p,
#         )
#         text = response.choices[0].message.content
#         self._save_cache(cache_key, text)
#         return text

# class AzureChatOpenAI(ChatOpenAI):
#     def __init__(
#         self,
#         model_string: str = "gpt-35-turbo",
#         system_prompt: str = ChatOpenAI.DEFAULT_SYSTEM_PROMPT,
#         **kwargs
#     ):
#         root = platformdirs.user_cache_dir("textgrad")
#         cache_path = os.path.join(root, f"cache_azure_{model_string}.db")
#         super().__init__(
#             model_string=model_string,
#             system_prompt=system_prompt,
#             **kwargs
#         )

#         api_key = os.getenv("AZURE_OPENAI_API_KEY")
#         api_base = os.getenv("AZURE_OPENAI_API_BASE")
#         api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")
#         if not api_key or not api_base:
#             raise ValueError(
#                 "Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_API_BASE environment vars."
#             )

#         self.client = AzureOpenAI(
#             api_key=api_key,
#             azure_endpoint=api_base,
#             azure_deployment=model_string,
#             api_version=api_version,
#         )
#         self.model_string = model_string
# -*- coding: utf-8 -*-
"""
Unified OpenAI / Azure-OpenAI 客户端封装
======================================

- ChatOpenAI  : 兼容官方 OpenAI、代理、Ollama、自托管等多种 base_url
- AzureChatOpenAI : Azure-OpenAI 专用，缓存与 ChatOpenAI 隔离
- 兼容环境变量：
    OPENAI_API_KEY     （必需）
    OPENAI_BASE_URL    （可选，优先级最高，旧约定）
    OPENAI_API_BASE    （可选，兼容某些平台）
    OLLAMA_BASE_URL    （可选，默认为 http://localhost:11434/v1）
    AZURE_OPENAI_API_KEY / BASE / VERSION
"""

from __future__ import annotations

import base64
import json
import os
from typing import List, Union

import platformdirs
from tenacity import retry, stop_after_attempt, wait_random_exponential

# --------------------------------------------------------------------------- #
# 第三方依赖
# --------------------------------------------------------------------------- #
try:
    from openai import OpenAI, AzureOpenAI
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "请先 `pip install openai`，并设置 OPENAI_API_KEY 环境变量！"
    ) from exc

# --------------------------------------------------------------------------- #
# 内部依赖
# --------------------------------------------------------------------------- #
from .base import EngineLM, CachedEngine
from .engine_utils import get_image_type_from_bytes

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _require_env(name: str) -> str:
    """读取环境变量；缺失时抛清晰错误。"""
    try:
        return os.environ[name]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"环境变量 {name} 未设置！") from exc


def _effective_base_url(explicit: str | None) -> str | None:
    """
    确定 base_url：
        1. 函数参数
        2. OPENAI_BASE_URL
        3. OPENAI_API_BASE
        4. None（官方端点）
    """
    return (
        explicit
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or None
    )


def _build_openai_client(api_key: str, base_url: str | None) -> OpenAI:
    """根据 base_url 分类构造 OpenAI 客户端。"""
    # ---- Ollama ----
    if base_url == OLLAMA_BASE_URL:
        return OpenAI(base_url=base_url, api_key="ollama")

    # ---- 其它自定义 / 代理 ----
    return OpenAI(api_key=api_key, base_url=base_url)


# --------------------------------------------------------------------------- #
# ChatOpenAI
# --------------------------------------------------------------------------- #
# class ChatOpenAI(EngineLM, CachedEngine):
#     """统一的 OpenAI 聊天接口封装。"""

#     DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

#     def __init__(
#         self,
#         model_string: str = "gpt-3.5-turbo-0613",
#         system_prompt: str = DEFAULT_SYSTEM_PROMPT,
#         is_multimodal: bool = False,
#         base_url: str | None = None,
#         **kwargs,
#     ):
#         # --- 缓存文件 ---
#         root = platformdirs.user_cache_dir("textgrad")
#         cache_path = os.path.join(root, f"cache_openai_{model_string}.db")
#         super().__init__(cache_path=cache_path)

#         # --- 属性 ---
#         self.model_string = model_string
#         self.system_prompt = system_prompt
#         self.is_multimodal = is_multimodal

#         # --- 客户端初始化 ---
#         api_key = _require_env("OPENAI_API_KEY")
#         self.base_url = _effective_base_url(base_url)
#         self.client: OpenAI = _build_openai_client(api_key, self.base_url)

#     # ------------------------------------------------------------------ #
#     # 对外 API
#     # ------------------------------------------------------------------ #
#     @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
#     def generate(
#         self,
#         content: Union[str, List[Union[str, bytes]]],
#         system_prompt: str | None = None,
#         **kwargs,
#     ):
#         if isinstance(content, str):
#             return self._generate_from_single_prompt(
#                 content, system_prompt=system_prompt, **kwargs
#             )

#         if isinstance(content, list):
#             if any(isinstance(x, bytes) for x in content) and not self.is_multimodal:
#                 raise NotImplementedError(
#                     "Multimodal 仅在 `is_multimodal=True` 时支持。"
#                 )
#             return self._generate_from_multiple_input(
#                 content, system_prompt=system_prompt, **kwargs
#             )

#         raise TypeError(f"Unsupported content type: {type(content)}")

#     def __call__(self, prompt: Union[str, List[Union[str, bytes]]], **kwargs):
#         return self.generate(prompt, **kwargs)

#     # ------------------------------------------------------------------ #
#     # 内部实现
#     # ------------------------------------------------------------------ #
#     def _generate_from_single_prompt(
#         self,
#         prompt: str,
#         system_prompt: str | None = None,
#         temperature: float = 0.0,
#         max_tokens: int = 4000,
#         top_p: float = 0.99,
#         stop: list[str] | None = None,
#     ):
#         sys_msg = system_prompt or self.system_prompt
#         cache_key = sys_msg + prompt
#         if (cached := self._check_cache(cache_key)) is not None:
#             return cached

#         resp = self.client.chat.completions.create(
#             model=self.model_string,
#             messages=[
#                 {"role": "system", "content": sys_msg},
#                 {"role": "user", "content": prompt},
#             ],
#             temperature=temperature,
#             max_tokens=max_tokens,
#             top_p=top_p,
#             frequency_penalty=0,
#             presence_penalty=0,
#             stop=stop,
#         )
#         answer = resp.choices[0].message.content
#         self._save_cache(cache_key, answer)
#         return answer

#     def _generate_from_multiple_input(
#         self,
#         content: List[Union[str, bytes]],
#         system_prompt: str | None = None,
#         temperature: float = 0.0,
#         max_tokens: int = 4000,
#         top_p: float = 0.99,
#         stop: list[str] | None = None,
#     ):
#         sys_msg = system_prompt or self.system_prompt
#         formatted = self._format_content(content)
#         cache_key = sys_msg + json.dumps(formatted, sort_keys=True)
#         if (cached := self._check_cache(cache_key)) is not None:
#             return cached

#         resp = self.client.chat.completions.create(
#             model=self.model_string,
#             messages=[
#                 {"role": "system", "content": sys_msg},
#                 {"role": "user", "content": formatted},
#             ],
#             temperature=temperature,
#             max_tokens=max_tokens,
#             top_p=top_p,
#             stop=stop,
#         )
#         answer = resp.choices[0].message.content
#         self._save_cache(cache_key, answer)
#         return answer

#     # --------
#     @staticmethod
#     def _format_content(content: List[Union[str, bytes]]) -> List[dict]:
#         """把文本 & 图片二进制封装成 OpenAI 所需格式。"""
#         formatted: list[dict] = []
#         for item in content:
#             if isinstance(item, bytes):
#                 img_type = get_image_type_from_bytes(item)
#                 b64 = base64.b64encode(item).decode()
#                 formatted.append(
#                     {
#                         "type": "image_url",
#                         "image_url": {"url": f"data:image/{img_type};base64,{b64}"},
#                     }
#                 )
#             elif isinstance(item, str):
#                 formatted.append({"type": "text", "text": item})
#             else:
#                 raise TypeError(f"Unsupported input type: {type(item)}")
#         return formatted

class ChatOpenAI(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

    def __init__(
        self,
        model_string: str = "gpt-3.5-turbo-0613",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
        base_url: str | None = None,
        **kwargs,
    ):
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_openai_{model_string}.db")
        super().__init__(cache_path=cache_path)

        self.model_string = model_string
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal

        api_key = _require_env("OPENAI_API_KEY")
        self.base_url = _effective_base_url(base_url)
        self.client: OpenAI = _build_openai_client(api_key, self.base_url)

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(
        self,
        content: Union[str, List[Union[str, bytes]]],
        system_prompt: str | None = None,
        return_full_response: bool = False,  # 新增参数
        **kwargs,
    ):
        if isinstance(content, str):
            return self._generate_from_single_prompt(
                content,
                system_prompt=system_prompt,
                return_full_response=return_full_response,
                **kwargs,
            )

        if isinstance(content, list):
            if any(isinstance(x, bytes) for x in content) and not self.is_multimodal:
                raise NotImplementedError(
                    "Multimodal 仅在 `is_multimodal=True` 时支持。"
                )
            return self._generate_from_multiple_input(
                content,
                system_prompt=system_prompt,
                return_full_response=return_full_response,
                **kwargs,
            )

        raise TypeError(f"Unsupported content type: {type(content)}")

    def __call__(self, prompt: Union[str, List[Union[str, bytes]]], **kwargs):
        return self.generate(prompt, **kwargs)

    def _generate_from_single_prompt(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        top_p: float = 0.99,
        stop: list[str] | None = None,
        return_full_response: bool = False,  # 新增参数
    ):
        sys_msg = system_prompt or self.system_prompt
        cache_key = sys_msg + prompt
        if (cached := self._check_cache(cache_key)) is not None and not return_full_response:
            return cached

        resp = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop,
        )
        if return_full_response:
            return resp
        else:
            answer = resp.choices[0].message.content
            self._save_cache(cache_key, answer)
            return answer

    def _generate_from_multiple_input(
        self,
        content: List[Union[str, bytes]],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        top_p: float = 0.99,
        stop: list[str] | None = None,
        return_full_response: bool = False,  # 新增参数
    ):
        sys_msg = system_prompt or self.system_prompt
        formatted = self._format_content(content)
        cache_key = sys_msg + json.dumps(formatted, sort_keys=True)
        if (cached := self._check_cache(cache_key)) is not None and not return_full_response:
            return cached

        resp = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": formatted},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
        )
        if return_full_response:
            return resp
        else:
            answer = resp.choices[0].message.content
            self._save_cache(cache_key, answer)
            return answer

    @staticmethod
    def _format_content(content: List[Union[str, bytes]]) -> List[dict]:
        formatted: list[dict] = []
        for item in content:
            if isinstance(item, bytes):
                img_type = get_image_type_from_bytes(item)
                b64 = base64.b64encode(item).decode()
                formatted.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{img_type};base64,{b64}"},
                    }
                )
            elif isinstance(item, str):
                formatted.append({"type": "text", "text": item})
            else:
                raise TypeError(f"Unsupported input type: {type(item)}")
        return formatted

# --------------------------------------------------------------------------- #
# AzureChatOpenAI
# --------------------------------------------------------------------------- #
class AzureChatOpenAI(ChatOpenAI):
    """Azure-OpenAI 专用封装，缓存独立。"""

    def __init__(
        self,
        model_string: str = "gpt-35-turbo",
        system_prompt: str = ChatOpenAI.DEFAULT_SYSTEM_PROMPT,
        **kwargs,
    ):
        # 先构造，以便父类校验 / 初始化
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_azure_{model_string}.db")

        # 调用父类（base_url 置 None，让其用官方逻辑；稍后替换 client）
        super().__init__(
            model_string=model_string,
            system_prompt=system_prompt,
            base_url=None,
            **kwargs,
        )

        # 覆写缓存路径（CachedEngine 在第一次 _check_cache 时才会落盘）
        self._cache_path = cache_path

        # Azure-OpenAI 专属环境变量
        api_key = _require_env("AZURE_OPENAI_API_KEY")
        endpoint = _require_env("AZURE_OPENAI_API_BASE")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")

        # 用 Azure SDK 客户端替换
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            azure_deployment=model_string,
            api_version=api_version,
        )
        self.base_url = endpoint  # 仅作记录
