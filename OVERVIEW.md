# ABD — 코드 구조 및 실험 프로세스

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [디렉토리 구조](#2-디렉토리-구조)
3. [pick_place 태스크](#3-pick_place-태스크)
4. [stack_cups 태스크](#4-stack_cups-태스크)
5. [open_drawer 태스크](#5-open_drawer-태스크)
6. [실험 실행 방법](#6-실험-실행-방법)

---

## 1. 프로젝트 개요

**ABD (Autonomy Boundary Detection)** 는 로봇 조작 데이터 자율 수집 시스템이다.  
PaPA (Pick-and-Place with Affordance) 파이프라인 위에서 동작하며, VLM 기반 체크리스트로 태스크 성공/실패를 판단하고 인간 리셋이 필요한 시점을 자동 결정한다.

**핵심 흐름:**
```
관측 → 궤적 생성 (VLM → Perception → Motion) → 실행 → 검증 → 리셋 정책 결정
```

**의존 관계:**
- `ABD/src/` 모듈이 `PaPA/src/` 모듈을 shadow(우선 덮어씀) — `bootstrap.py` 에서 sys.path 정렬

---

## 2. 디렉토리 구조

```
ABD/
├── scripts/
│   ├── main.py                  # Hydra 진입점
│   ├── run_all_policies.sh      # 여러 정책 일괄 비교 실험
│   ├── run_aha_server.py        # AHA VLM 서버 실행
│   ├── run_qwen_server.py       # Qwen VLM 서버 실행
│   ├── analyze_checklist.py     # 체크리스트 결과 분석
│   └── analyze_ground_truth.py  # Ground truth 라벨 분석
│
├── config/
│   ├── config.yaml              # 최상위 default 설정
│   ├── env/                     # real.yaml / dummy.yaml
│   ├── task/                    # pick_place.yaml / stack_cups.yaml / open_drawer.yaml
│   ├── policy/                  # naive / no_reset / periodic / vlm_checklist
│   ├── vlm/                     # gpt.yaml / qwen.yaml / aha.yaml
│   ├── dataset_recorder/        # LeRobot.yaml
│   └── checklists/              # 태스크별 VLM 체크리스트 JSON (런타임 생성)
│
└── src/
    ├── bootstrap.py             # sys.path 설정 (ABD > PaPA 우선순위)
    ├── prompts.py               # VLM_PROMPT_TEMPLATE
    ├── runner/
    │   ├── collection_runner.py # 메인 수집 루프 (핵심 오케스트레이터)
    │   └── human_reset_interface.py
    ├── task/
    │   ├── task_family.py       # TaskDefinition / ReversibleTaskPair / TaskSequence / TaskScheduler
    │   └── task_registry.py     # build_*_task_from_config() 팩토리
    ├── trajectory_generator/
    │   ├── papa_generator.py    # VLM Planner → Perception → Motion (실제 하드웨어)
    │   └── dummy_generator.py   # dummy 환경 전용
    ├── executor/
    │   └── trajectory_executor.py
    ├── validator/
    │   ├── vlm_validator.py     # VLM 기반 성공 검증 (VLMChecklist 외 정책의 fallback)
    │   └── dummy_validator.py   # dummy 환경 전용
    ├── policy/
    │   ├── vlm_checklist_policy.py  # 메인 정책
    │   ├── naive_policy.py
    │   ├── periodic_policy.py
    │   └── no_reset_policy.py
    ├── vlm/
    │   ├── vlm_planner.py       # pick/place/action 파싱
    │   └── vlm_input.py
    ├── vlm_client/
    │   ├── vqa_client.py        # VQA (이미지+텍스트 질의) 래퍼
    │   ├── factory.py           # 백엔드 팩토리
    │   └── backends/            # gpt.py / qwen.py / aha.py
    ├── perception/
    │   ├── perception_agent.py  # PaPA 인식 에이전트 래퍼
    │   └── perception_input.py
    ├── motion_planner/
    │   └── motion_planner.py    # AprilTag 기반 IK/웨이포인트 생성
    ├── env/
    │   ├── base_env.py
    │   ├── real_env.py          # Piper 로봇 + RealSense 카메라
    │   └── dummy_env.py
    ├── robot/
    │   └── piper.py             # Piper SDK 래퍼
    ├── camera/
    │   └── realsense.py         # RealSense D4xx 래퍼
    ├── dataset/
    │   └── dataset_recorder.py  # LeRobot 포맷 데이터셋 기록
    ├── metrics/
    │   ├── metrics_logger.py    # EpisodeRecord, InterventionRecord, 집계 통계
    │   ├── failure_classifier.py
    │   └── dataset_manifest.py
    └── utils/
        ├── camera_utils.py      # AprilTag 감지 유틸
        ├── transform_utils.py
        ├── trajectory_utils.py
        └── obb_utils.py
```

---

## 3. pick_place 태스크

### 설정

```yaml
# config/task/pick_place.yaml
family: pick_place_reversible
object: "banana"
location_a: "plate"
location_b: "pan"
forward_task: "pick the banana and put it in the pan"
reverse_task: "pick the banana and put it in the plate"
```

### 실행 사이클

```
forward: banana → pan   (banana_to_pan)
reverse: banana → plate (banana_to_plate)
```

에피소드 성공 시 forward ↔ reverse를 번갈아 실행한다.  
reset이 발생하면 항상 forward(banana→pan)로 돌아간다.

### 궤적 생성 (`_generate_pick_place`)

```
1. VLMPlanner.query()
   → 현재 카메라 이미지로 pick 객체명, place 객체명, action 방향 파싱
   → 최대 10회 재시도 (pick/place 둘 다 확인될 때까지)

2. PerceptionAgent.query(pick_object)
   → 외부 inference 서버(port 9877)로 pick 객체 3D 위치 요청
   → 최대 10회 재시도

3. PerceptionAgent.query(place_object)
   → place 객체 3D 위치 요청

4. MotionPlanner.plan_pick_place()
   → pick/place pose + place_offset 적용하여 웨이포인트 생성
   → place 객체에 "pan"이 포함되면 bowl-center XY 보정 적용
```

생성 실패 처리:

| 실패 원인 | `reason` 값 | 처리 |
|---|---|---|
| VLM이 객체 식별 실패 | `vlm_failed` | human reset 요청 |
| pick 객체 인식 실패 | `pick_perception_failed` | VLM 체크리스트로 라우팅 |
| place 객체 인식 실패 | `place_perception_failed` | VLM 체크리스트로 라우팅 |

### 검증 및 에피소드 결정

실행 후 init pose로 복귀, `VLMChecklistPolicy.needs_reset()` 호출:

```
Phase 1: common + current_task 평가 → success 여부 결정
Phase 2: success이면 짝 태스크의 next_task / 실패이면 현재 태스크의 next_task 평가

score = Σ(weight_i × yes_i) / Σ(weight_i)

score >= τ_reset(0.9)  → next  (성공, 다음 스텝으로)
score <  τ_reset       → reset (환경 리셋 요청)
success=False & reset 불필요 → retry (페이즈 시작으로 되돌림)
```

### 체크리스트

| 파일 | 태스크 | 성공 판단 기준 |
|---|---|---|
| `banana_to_pan.json` | forward | "Is the banana currently placed inside the pan?" |
| `banana_to_plate.json` | reverse | "Is the banana currently placed on the plate?" |

짝 관계: `banana_to_pan` ↔ `banana_to_plate`

체크리스트 항목 구조 (banana_to_pan 기준):

| 섹션 | 항목 수 | 주요 내용 |
|---|---|---|
| `common` | 5 | 로봇 안전, 그리퍼, 인식 시스템, 바나나 가시성/도달 가능성 |
| `current_task` | 3 | 바나나가 팬 안에 있는지(success_item), 바나나·팬 인식 여부 |
| `next_task` | 5 | 팬 가시성, 팬 접근성, 놓을 공간 확보 |

---

## 4. stack_cups 태스크

### 설정

```yaml
# config/task/stack_cups.yaml
family: stack_cups
cup_a: "purple"   tag_id_a: 5   # 첫 번째 pick 컵
cup_b: "blue"     tag_id_b: 4   # 베이스 컵
cup_c: "pink"     tag_id_c: 6   # 두 번째 pick 컵
```

컵 위치는 AprilTag(ID 4, 5, 6)로 직접 감지한다. VLMPlanner/PerceptionAgent는 사용하지 않는다.

### 실행 사이클 (4-step)

```
forward_1: purple → blue 위에 쌓기         [비터미널 — VLM 검증 없이 auto_advance]
forward_2: pink → purple 위에 쌓기         [터미널   — VLM 체크리스트 평가]
    ↓ reset 발생 시 forward_1으로 복귀
reverse_1: pink → 원래 위치로 되돌리기     [비터미널 — VLM 검증 없이 auto_advance]
reverse_2: purple → 원래 위치로 되돌리기   [터미널   — VLM 체크리스트 평가]
    ↓ reset 발생 시 forward_1으로 복귀
```

- **비터미널 스텝**: 실행 성공 시 체크리스트 없이 바로 다음 스텝으로 진행
- **터미널 스텝**: 체크리스트 평가 결과로 next / retry / reset 결정
- retry 발생 시: 현재 phase 시작(forward_1 또는 reverse_1)으로 되돌아감

### 궤적 생성 (`_generate_stack_cups`)

```
MotionPlanner.plan_stack_cups(pick_tag_id, place_tag_id, place_xy_offset, stack_step)
```

- **forward 스텝**: pick/place 태그 모두 실시간 감지
- **reverse 스텝**: `place_xy_offset=[0,0]` → `_tag_position_cache`에서 forward 시점에 저장된 원래 위치 조회  
  (컵이 쌓여 태그가 가려져도 동작)

태그 감지 실패(`reason="apriltag_not_found"`) 시: 비터미널/터미널 모두 VLM 체크리스트로 라우팅  
(pick/place 어느 쪽 감지 실패인지 `pick_detected`, `place_detected`로 기록 → 체크리스트 detection_info에 전달)

### 데이터셋 저장

- forward_1 프레임과 forward_2 프레임은 버퍼에 누적 → forward_2 터미널 완료 시 한꺼번에 저장
- reverse도 동일 (reverse_1 누적 → reverse_2에서 저장)
- retry/reset 시 버퍼 클리어 후 해당 phase 처음부터 재수집

### 검증 및 에피소드 결정

터미널 스텝(forward_2, reverse_2)에서만 `VLMChecklistPolicy.needs_reset()` 호출.  
비터미널 스텝은 `validation = ValidationResult(success=True, method="auto_advance")`로 처리.

### 체크리스트

| 파일 | 태스크 | 성공 판단 기준 |
|---|---|---|
| `stack_cups_forward.json` | forward_2 | purple이 blue 위에 AND pink이 purple 위에 (둘 다 yes) |
| `stack_cups_reverse.json` | reverse_2 | purple이 blue 위에 AND pink이 purple 위에 (둘 다 no) |

짝 관계: `stack_cups_forward` ↔ `stack_cups_reverse`

`success_answer` 및 `reset_group`:

| 항목 | forward | reverse |
|---|---|---|
| "Is the purple cup stacked on the blue cup?" | `yes` | `no` (weight=-0.5) |
| "Is the pink cup stacked on the purple cup?" | `yes` | `no` (weight=-0.5) |
| reset_group | 1 | 1 |

`reset_group` 동작: 같은 그룹 내 답이 혼재(예: 하나는 yes, 하나는 no)하면 해당 그룹 점수를 0으로 처리 → 부분 성공 상태에서 false positive 방지

체크리스트 항목 구조:

| 섹션 | 항목 수 | 주요 내용 |
|---|---|---|
| `common` | 8 | 로봇/그리퍼/센서 안전, 컵 쓰러짐 여부, 작업 공간 내 위치, 경로 확보 |
| `current_task` | 4 | 쌓임/분리 여부(success_item ×2), pick·place 컵 인식 여부 |
| `next_task` | 1 | 컵들이 안정적인 표면에 있는지 |

### `_stack_cups_prev_detection` 캐시

reverse_1 완료 시 pink cup 감지 결과를 `_stack_cups_prev_detection`에 저장해둔다.  
reverse_2에서 purple 태그가 감지 실패해도 이 캐시로 detection_info를 대체한다.  
(pink이 아직 purple 위에 있으면 purple 태그가 보이지 않아 오탐이 발생할 수 있으므로)

---

## 5. open_drawer 태스크

### 설정

```yaml
# config/task/open_drawer.yaml
family: open_drawer
tag_id: 7   # 서랍 손잡이에 부착된 AprilTag ID
```

### 실행 사이클

```
forward: 서랍 열기 (손잡이를 -X 방향으로 pull)
reverse: 서랍 닫기 (손잡이를 +X 방향으로 push)
```

reset이 발생하면 항상 forward(열기)로 돌아간다.

### 궤적 생성 (`_generate_open_drawer`)

```
MotionPlanner.plan_open_drawer(pick_tag_id=7, stack_step="forward"/"reverse")
→ AprilTag(tag_id=7) 감지로 손잡이 위치 결정
→ stack_step에 따라 pull/push 방향 결정
→ 생성 성공 시 last_drawer_rotation_deg를 metadata["drawer_rotation_deg"]에 기록
```

태그 감지 실패(`reason="apriltag_not_found"`) 시: human reset 요청  
(stack_cups와 달리 체크리스트로 라우팅하지 않음)

### 서랍 회전 기준각 (`_drawer_ref_yaw`)

```
최초 forward 실행 전:
  _drawer_ref_yaw가 None이면 measure_drawer_rotation(tag_id) 호출
  → 정면 기준 yaw를 _drawer_ref_yaw에 저장

매 validation 전:
  현재 rotation을 재측정, 기준각 대비 회전량(drawer_rotation_deg) 계산
  → detection_info["drawer rotation"] = "X.X degrees left/right"로 체크리스트에 전달

reset 발생 시:
  reset_drawer_reference() 호출 → _drawer_ref_yaw = None (다음 forward에서 재측정)
```

### 검증 및 에피소드 결정

실행 후 init pose로 복귀, `VLMChecklistPolicy.needs_reset()` 호출.  
평가 방식은 pick_place와 동일 (Phase 1 → Phase 2 → score 계산).

카메라 입력: `front` 카메라 이미지를 제외하고 wrist + table 카메라만 체크리스트 평가에 사용.

### 체크리스트

| 파일 | 태스크 | 성공 판단 기준 |
|---|---|---|
| `open_drawer.json` | forward | "Is the drawer open far enough that the interior is clearly visible?" |
| `close_drawer.json` | reverse | "Is the drawer closed so that none of their interiors are visible?" |

짝 관계: `open_drawer` ↔ `close_drawer`

체크리스트 항목 구조 (open_drawer 기준):

| 섹션 | 항목 수 | 주요 내용 |
|---|---|---|
| `common` | 8 | 로봇/그리퍼/센서 안전, 손잡이 가시성/도달 가능성, 레일 상태 |
| `current_task` | 3 | 서랍이 충분히 열렸는지(success_item), AprilTag 감지 여부, 회전각 15° 이내 여부 |
| `next_task` | 2 | 손잡이 접근성, 손잡이 파손 여부 |

---

## 6. 실험 실행 방법

### 사전 준비

```bash
# 1. CAN 인터페이스 활성화 (Piper 로봇)
bash ~/Desktop/codes/piper_sdk/piper_sdk/can_activate.sh

# 2. Perception 서버 실행
bash ~/Desktop/codes/PaPA/run_DINO_server.sh

# 3. VLM 서버 (로컬 모델 사용 시)
python scripts/run_qwen_server.py   # Qwen
python scripts/run_aha_server.py    # AHA
```

### 실험 실행

```bash
cd ~/Desktop/codes/ABD

# 기본 실행 (pick_place, VLMChecklist, GPT)
python scripts/main.py

# 태스크 지정
python scripts/main.py task=stack_cups
python scripts/main.py task=open_drawer

# 정책 변경
python scripts/main.py policy=naive
python scripts/main.py policy=no_reset



### 실험 후 분석

```bash
# 체크리스트 결과 분석
python scripts/analyze_checklist.py outputs/logs/<run_dir>/

# ground truth 라벨 분석
python scripts/analyze_ground_truth.py outputs/logs/<run_dir>/
```

### 출력 파일

```
outputs/logs/<YYYYMMDD_HHMMSS>/
├── run_config.json         # Hydra 설정 스냅샷
├── episodes.jsonl          # 에피소드별 레코드 (1줄=1에피소드)
├── intervention_log.jsonl  # human reset 이벤트
└── summary.json            # 집계 통계 (success_rate, interventions_per_100_valid 등)
```

데이터셋 디렉토리: `run_YYYYMMDD_N`  
(같은 날 첫 번째 실행은 `run_20260428_0`, 두 번째는 `run_20260428_1`, ...)
