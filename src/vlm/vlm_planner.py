import logging
import re
from typing import Dict, Tuple, Optional

import numpy as np

from vlm_client.base import VLMBackend, VLMRequest
from prompts import VLM_PROMPT_TEMPLATE

log = logging.getLogger(__name__)


class VLMPlanner:
    def __init__(self, backend: VLMBackend):
        self.backend = backend

    def query(
        self,
        observation: Dict[str, np.ndarray],
        task: str,
    ) -> Tuple[Optional[str], Optional[str], str]:
        prompt = VLM_PROMPT_TEMPLATE.format(TASK_DESCRIPTION=task)
        response = self.backend.generate(
            VLMRequest(prompt=prompt, images=observation, task_kind="plan")
        )
        log.debug(f"[VLMPlanner] raw response:\n{response}")
        plan = self.parse_response_to_plan(response)
        pick, place, action = plan
        log.info(f"[VLMPlanner] parsed → pick={pick}, place={place}, action={action}")

        return plan

    def parse_response_to_plan(self, response: str) -> Tuple[Optional[str], Optional[str], str]:
        # Qwen3-VL produces responses with a different conversational shape
        # (line-prefixed "PICK object:" / "Action direction:" formats), so
        # we route to a Qwen-specific parser when that backend is active.
        if getattr(self.backend, "name", "") == "qwen":
            action = self._parse_action_qwen(response)
            pick_object = self._parse_reference_object_qwen(response, tag='pick')
            place_object = self._parse_reference_object_qwen(response, tag='place')
        else:
            action = self._parse_action(response)
            pick_object = self._parse_reference_object(response, tag='pick')
            place_object = self._parse_reference_object(response, tag='place')

        return pick_object, place_object, action

    def _parse_action(self, response, action_list=['front','back','left','right','up','down'], default='front'):
        response = response.lower()
        tag_patterns = [
            r"[<\[\(]\s*action\s*[>\]\)](.*?)[<\[\(]\s*/\s*action\s*[>\]\)]"
        ]
        candidates = []

        for pattern in tag_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for m in matches:
                candidates.append(m.strip())

        if not candidates:
            for a in action_list:
                if re.search(rf"\b{a}\b", response):
                    candidates.append(a)

        if not candidates:
            return default

        final_candidate = candidates[-1]

        for a in action_list:
            if re.search(rf"\b{a}\b", final_candidate):
                return a

        return default

    def _strip_reasoning_block(self, text: str) -> str:
        # <reasoning>...</reasoning> 제거 (대소문자/공백 변형도 허용)
        return re.sub(
            r"<\s*reasoning\s*>.*?<\s*/\s*reasoning\s*>",
            " ",
            text,
            flags=re.DOTALL | re.IGNORECASE
        )

    def _find_tag_contents(self, text: str, tag: str):
        """
        1) 줄 시작에 있는 <tag>...</tag> 블록을 최우선으로 찾고
        2) 없으면 일반 <tag>...</tag>를 찾는다.
        - 너무 긴 캡처(= reasoning부터 걸친 경우)나 다른 주요 태그 포함되면 버림
        """
        candidates = []

        # (A) 줄 시작(멀티라인) 블록 형태: 네 출력 예시(<pick>\n...\n</pick>)에 가장 안전
        pat_lineblock = re.compile(
            rf"(?ims)^\s*<\s*{re.escape(tag)}\s*>\s*(.*?)\s*^\s*<\s*/\s*{re.escape(tag)}\s*>\s*$"
        )
        for m in pat_lineblock.findall(text):
            s = m.strip()
            candidates.append(s)

        # (B) 인라인 형태도 지원: <pick>brown food</pick>
        if not candidates:
            pat_inline = re.compile(
                rf"(?is)<\s*{re.escape(tag)}\s*>\s*(.*?)\s*<\s*/\s*{re.escape(tag)}\s*>"
            )
            for m in pat_inline.findall(text):
                s = m.strip()
                candidates.append(s)

        # (C) “reasoning부터 걸친 거대한 매치” 같은 이상치 제거
        cleaned = []
        for c in candidates:
            # 너무 길면(대부분 reasoning~정답까지 걸친 매치) 버림
            if len(c) > 120:
                continue
            # 다른 섹션 태그/키워드가 섞이면 버림
            if "<reasoning" in c or "</reasoning" in c:
                continue
            if "pick object" in c or "place object" in c or "action direction" in c:
                continue
            cleaned.append(c)

        return cleaned

    def _normalize_object_token(self, obj: str, keep_underscore: bool, max_scan_words: int):
        obj = obj.strip()
        if keep_underscore:
            obj = re.sub(r"[^a-z0-9_\s]", " ", obj)
        else:
            obj = obj.replace("_", " ")
            obj = re.sub(r"[^a-z0-9\s]", " ", obj)

        obj = re.sub(r"\s+", " ", obj).strip()
        obj = re.sub(r"^(the|a|an|this|that)\s+", "", obj).strip()
        if not obj:
            return None

        stop_words = {
            "on", "in", "at", "near", "next", "to", "under", "over", "above", "below", "beside",
            "left", "right", "front", "back", "top", "bottom", "table", "robot", "of", "with"
        }
        words = obj.split()[:max_scan_words]
        chunk = []
        for w in words:
            if w in stop_words:
                break
            chunk.append(w)

        token = (chunk[-1] if chunk else (words[0] if words else None))
        return token

    def _parse_reference_object(
            self,
            response: str,
            *,
            allowed_objects=None,
            default=None,
            keep_underscore=False,
            max_scan_words=12,
            tag='reference_object',
    ):
        text = (response or "").lower()

        # 핵심: reasoning 제거 후 파싱
        text = self._strip_reasoning_block(text)

        # 1) 태그 기반 후보 추출 (라인블록 우선)
        candidates = self._find_tag_contents(text, tag)

        if not candidates:
            return default

        # 여러 개면 마지막
        obj = candidates[-1]

        # 2) 기본 정규화 + 단일 토큰화
        token = self._normalize_object_token(obj, keep_underscore=keep_underscore, max_scan_words=max_scan_words)
        if not token:
            return default

        # 3) allowed_objects가 있으면 여기서 추가 매칭 로직을 붙일 수 있음(원하면 확장)
        return token or default

    def _parse_action_qwen(
            self,
            response: str,
            action_list=('front', 'back', 'left', 'right', 'up', 'down'),
            default='front'
    ) -> str:
        text = (response or "").lower()

        # reasoning 블록은 노이즈가 많아서 제거(최종 답은 보통 밖에 따로 있음)
        text = re.sub(r"<reasoning>.*?</reasoning>", " ", text, flags=re.DOTALL | re.IGNORECASE)

        # action 동의어 조금만 보정
        synonym = {
            "forward": "front", "forwards": "front", "ahead": "front",
            "backward": "back", "backwards": "back",
            "upward": "up", "downward": "down",
        }

        candidates = []

        # 1) <action>...</action>
        for m in re.findall(r"<\s*action\s*>(.*?)<\s*/\s*action\s*>", text, flags=re.DOTALL | re.IGNORECASE):
            candidates.append(m.strip())

        # 2) "Action direction (WRIST frame): right" / "Action direction ...:\nright"
        #    콜론 뒤 한 줄(또는 다음 줄)에서 후보 뽑기
        #    - 너무 길게 잡지 않도록 1줄만
        pat_dir_line = re.compile(
            r"action\s*direction.*?:\s*(?:\n\s*)?([^\n\r<>'\"]+)",
            flags=re.IGNORECASE
        )
        for m in pat_dir_line.findall(text):
            candidates.append(m.strip())

        # 3) 최후의 fallback: 텍스트 전체에서 방향 단어 등장한 순서대로 수집
        all_dirs = re.findall(r"\b(front|back|left|right|up|down|forward|backward|ahead)\b", text)
        candidates += all_dirs

        # 후보가 없으면 default
        if not candidates:
            return default

        # 마지막 후보를 우선 (여러 샘플이 ' / '로 붙어도 마지막이 최신인 경우가 많음)
        final = candidates[-1].strip().strip("\"'`[](){}<>")
        final = synonym.get(final, final)

        # final에서 방향 단어 추출
        for a in action_list:
            if re.search(rf"\b{re.escape(a)}\b", final):
                return a

        # final이 문장이라면, 그 문장 안에서 방향 단어를 다시 찾기
        for a in action_list:
            if re.search(rf"\b{re.escape(a)}\b", text):
                # 마지막 등장 방향을 고르기 위해 findall 후 마지막 사용
                hits = re.findall(rf"\b{re.escape(a)}\b", text)
                if hits:
                    # 전체 후보 중 마지막 방향을 찾는 방식이므로
                    pass

        # 전체 텍스트에서 방향 단어를 끝에서부터 찾기
        dir_hits = re.findall(r"\b(front|back|left|right|up|down)\b", text)
        return dir_hits[-1] if dir_hits else default

    def _parse_reference_object_qwen(
            self,
            response: str,
            *,
            allowed_objects=None,
            default=None,
            keep_underscore=True,
            max_scan_words=12,
            tag='reference_object',
    ) -> Optional[str]:
        """
        tag='pick' or 'place'일 때:
          - <pick>...</pick> / <place>...</place> 도 지원
          - "PICK object:" / "PLACE object or location:" 라인 기반 포맷도 지원
        """
        text = (response or "").lower()
        text = re.sub(r"<reasoning>.*?</reasoning>", " ", text, flags=re.DOTALL | re.IGNORECASE)

        candidates = []

        # A) 태그 기반 (<pick>...</pick>, <place>...</place>)
        tag_patterns = [
            fr"<\s*{re.escape(tag)}\s*>(.*?)<\s*/\s*{re.escape(tag)}\s*>",
            # 깨진 태그/괄호형까지 약간 허용
            fr"[<\[\(]\s*{re.escape(tag)}\s*[>\]\)](.*?)[<\[\(]\s*/\s*{re.escape(tag)}\s*[>\]\)]"
        ]
        for pat in tag_patterns:
            for m in re.findall(pat, text, flags=re.DOTALL | re.IGNORECASE):
                candidates.append(m.strip())

        # B) 라인 기반: "pick object:" / "place object or location:"
        #    tag가 pick/place일 때만 의미있게 동작
        if tag in ("pick", "place"):
            # 예: "1) pick object:\nbanana" / "place object or location:\nplate"
            line_pat = re.compile(
                rf"{tag}\s*object(?:\s*or\s*location)?\s*:\s*(?:\n\s*)?([^\n\r<>'\"]+)",
                flags=re.IGNORECASE
            )
            for m in line_pat.findall(text):
                candidates.append(m.strip())

        if not candidates:
            return default

        # 여러 개면 마지막(보통 최종 정답 섹션이 아래에 있음)
        obj_raw = candidates[-1]

        # 정규화
        obj = obj_raw.strip()
        obj = obj.strip("\"'`[](){}<>")

        if keep_underscore:
            # underscore 유지 + 영숫자/underscore/space만 남김
            obj = re.sub(r"[^a-z0-9_\s-]", " ", obj)
        else:
            obj = obj.replace("_", " ")
            obj = re.sub(r"[^a-z0-9\s-]", " ", obj)

        obj = obj.replace("-", " ")
        obj = re.sub(r"\s+", " ", obj).strip()
        obj = re.sub(r"^(the|a|an|this|that)\s+", "", obj).strip()

        if not obj:
            return default

        # allowed_objects가 있으면 가장 잘 맞는 걸 고르는 게 제일 정확
        if allowed_objects:
            # 공백/underscore 모두 대응
            norm = lambda s: re.sub(r"\s+", " ", s.lower().replace("_", " ").strip())
            obj_norm = norm(obj)
            allowed_norm = [(o, norm(o)) for o in allowed_objects]

            # 1) 완전일치 우선
            for o, on in allowed_norm:
                if obj_norm == on:
                    return o

            # 2) 포함 매칭(가장 긴 매치 우선)
            hits = []
            for o, on in allowed_norm:
                if on and on in obj_norm:
                    hits.append((len(on), o))
            if hits:
                hits.sort(reverse=True)
                return hits[0][1]

        # allowed_objects 없거나 매칭 실패면: 관계어 전에 나온 명사(마지막 단어)를 택함
        stop_words = {
            "on", "in", "at", "near", "next", "to", "under", "over", "above", "below", "beside",
            "left", "right", "front", "back", "top", "bottom", "table", "robot", "of", "with",
            "wrist", "frame", "direction", "object", "location"
        }
        words = obj.split()[:max_scan_words]
        chunk = []
        for w in words:
            if w in stop_words:
                break
            chunk.append(w)

        if not chunk:
            token = words[0] if words else ""
        else:
            token = chunk[-1]

        token = token.strip()
        if keep_underscore:
            token = token.replace(" ", "_")
        return token or default