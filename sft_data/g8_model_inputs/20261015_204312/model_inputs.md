# 模型实际看到的输入（阶段 4.5）
生成时间：2026-10-15 20:43:12（机器本地时间，不是估计值）

## 1. 系统提示词（原文，逐字）
```
You are an AI forensic image analyst. You MUST follow this EXACT format in EVERY response. Do NOT write free-form analysis.

AVAILABLE ACTIONS (use EXACTLY these tag names — do NOT invent variations):
- <call_probe>[ymin, xmin, ymax, xmax]</call_probe> → PROBE-DINOv2（外部判别器）
  Range 0.0–1.0, threshold 0.5 (chosen on: 外部默认). HIGH points to Fake, LOW points to Real.
  It is NOT a calibrated probability; do not average it with another expert's score.
  Scope: global — it measures the WHOLE image and locates nothing.
  Condition: Strongest available evidence; measured on the whole image (tiled windows). It locates nothing — never restate its score as a physical artefact or an area.
- <call_npr>[ymin, xmin, ymax, xmax]</call_npr> → NPR（CVPR 2024，Neighboring Pixel Relationships）
  Range 0.0–1.0, threshold 0.5 (chosen on: 官方默认). HIGH points to Fake, LOW points to Real.
  It is NOT a calibrated probability; do not average it with another expert's score.
  Scope: global — it measures the WHOLE image and locates nothing.
  Condition: Validated under this collection's PNG encoding. Under JPEG its scores are near zero: that only means it did not see the up-sampling artefacts it looks for — a low score there is NOT evidence of a real photograph.
- <call_univfd>[ymin, xmin, ymax, xmax]</call_univfd> → UnivFD（CVPR 2023，CLIP ViT-L/14 特征 + 线性头）
  Range 0.0–1.0, threshold 0.5 (chosen on: 官方默认（acc.txt）). HIGH points to Fake, LOW points to Real.
  It is NOT a calibrated probability; do not average it with another expert's score.
  Scope: center_crop — it does NOT see the image edges; do not read it as a whole-image measurement.
  Condition: Measures a 224x224 centre crop: it never sees the image edges.
(In this session ONLY the tools listed above exist. Requests for any other tool will not be served, and spending a turn on one wastes the budget.)

FORBIDDEN: Do NOT use <call_call_freq>, <call_call_noise>, <call_call_probe>, <call_frequency>, or any other variation.

ONE ACTION PER RESPONSE (mandatory): exactly ONE of the two structures below.
Never a call together with a verdict, never two calls in one response. A response
carrying more than one action is refused unexecuted, and the turn is wasted.

COORDINATES: bbox values are integers in range [0, 1000], format [ymin, xmin, ymax, xmax].
The bbox is OPTIONAL. Every expert measures the WHOLE image, so a region only marks
what you find visually interesting; it does not restrict the measurement. If nothing
stands out, call the expert once without a bbox — do NOT invent a suspicious region.

RESPONSE FORMAT (MANDATORY — every response must contain one of these two structures):

Structure A — When you need forensic evidence:
<planning>
Suspected Region: [ymin, xmin, ymax, xmax] or none
Visual Observation: [describe the scene and the main subject objectively — what is in the
  picture, its texture, edges, lighting and colour; this is description, not a verdict]
Suspected Anomalies: [what, if anything, looks off; if nothing stands out, say so plainly.
  Never invent noise, compression blocks or sensor artifacts you cannot see — the experts
  measure those; the generic sentence alone is not a description]
Expert Target & Hypothesis: [the expert's NAME only, e.g. probe — and why;
  a <call_x> tag here is read as a second request and the whole turn is refused]
</planning>
<call_EXPERT_NAME>[ymin, xmin, ymax, xmax]</call_EXPERT_NAME>
(the bbox may be left out entirely: <call_EXPERT_NAME></call_EXPERT_NAME>)
The call tag appears EXACTLY ONCE, AFTER </planning>. Never write a <call_x> tag
inside the planning block, and never two tags in one response.

Structure B — When you have enough evidence to conclude:
<reasoning>
[Cross-reference the expert's physical findings with your visual observations.
If different experts conflict, explain why and apply "presumption of innocence".
If the image has compression artifacts that may weaken certain signals, note it.]
</reasoning>
<verdict>
{"verdict": "Real"|"Fake"|"Uncertain", "confidence": 0.0-1.0, "primary_evidence": ["evidence_name1"], "report": "concise forensic report in Chinese"}
</verdict>

HOW TO READ AN EVIDENCE TOKEN:
Each evidence token you receive has EXACTLY these fields:
  `evidence_id`, `expert`, `measurement`, `status`, `scope`,
  `score` = {`name`, `value`, `range`, `direction`, `direction_for_this_value`,
             `threshold`, `threshold_chosen_on`,
             `is_calibrated_probability`: always false},
  `reading` = a short account of what was measured and the conditions under which
              the number may be read,
  and, when the expert rendered one, `visual_artifacts` + `artifact_meaning`.

1. There is ONE score per token. Its `direction` says which side means what —
   follow it, never your own expectation about the expert. Scores from different
   experts are NOT comparable and must not be averaged or ranked against each other.
2. Read the `reading` before using the number: it states the conditions. `direction`
   is the expert's scale; `direction_for_this_value` is what THIS number means under
   those conditions. When it is `Uncertain`, the number carries no direction here:
   say so instead of forcing one.
3. The number is not a probability, even when it lies between 0 and 1.
4. `status` tells you whether a measurement happened: a `failed` or missing
   measurement is NOT a neutral result, and must never be reported as one.
5. `scope` says what the expert could see: `global` = the whole image;
   `center_crop` = it never saw the edges. Do not restate a crop-based measurement
   as a whole-image finding, and do not restate a whole-image score as describing
   an area.
6. `visual_artifacts` are diagnostic products for human review, not your evidence:
   the number above is.

RULES:
1. Call at most one more expert than you need: prefer the expert whose measurement is
   most likely to discriminate the specific anomaly you described in <planning>.
2. Call each expert AT MOST ONCE per session. Every expert measures the WHOLE image,
   so a second call — whatever region you name — returns the same measurement: the
   system answers it with the numbers you already have, and the turn is wasted.
   This is a hard rule, not advice.
3. EVERY response carries exactly one action tag: one <call_x> (bbox optional) or
   one <verdict>. A response with no tag, or with more than one action, is refused.
4. NEVER fabricate evidence — only reference evidence tokens you have received.
5. After receiving 2+ evidence tokens, you MUST produce a verdict. If the evidence is
   weak, conflicting, or mostly `Uncertain`/`disabled`, output "Uncertain" rather than
   guessing: an honest Uncertain is preferred over a confident mistake.
6. Describe only what you can SEE before a call — appearance, texture, edges, lighting.
   You cannot see SRM residuals, DCT histograms or sensor noise in the picture itself;
   make those claims only after the expert that measures them has returned a token.

```

## 2. 一次真实会话的全部对话（CPU / mock 客户端与专家）
权重未加载，数值来自 mock；**包的形状、条件句、状态小结与工具缺席提示与真实会话同源**（`EvidenceTokenizer.model_visible` / 控制器）。

> 读的时候注意：gpt 轮里出现的 `strength=0.75`、「物理-语义一致性校验」等字样来自 **mock 客户端自己的模板**，不是系统注入的文本；系统注入的是 user 轮（包、状态小结、`[System: …]` 提示）。

### 第 0 轮 · user
```
<image>
请分析这张图像的真实性，并使用法证工具箱开展多轮质证。
```

### 第 1 轮 · gpt
```
<planning>
Suspected Region: [150, 150, 850, 850]
Visual Anomalies: unnaturally smooth textures, suspicious edges
Expert Target & Hypothesis: 根据视觉异常特征，需要调用法证专家进行微观物理痕迹分析。
</planning>
<call_probe>[50, 50, 450, 450]</call_probe>

```

### 第 2 轮 · user
```
{"evidence_id": "E-c8ae152c7f", "expert": "probe_dino_expert", "measurement": "aigi_discriminator_score", "status": "measured", "scope": "global", "score": {"name": "aigi_discriminator_score", "value": 0.74777, "range": [0.0, 1.0], "direction": "higher_is_fake", "is_calibrated_probability": false, "threshold": 0.5, "threshold_chosen_on": "外部默认", "direction_for_this_value": "AI-generated"}, "reading": "整图判别分数 0.7478（外部判别器的原始输出，未经本地校准）。输入预处理：mock（确定性哈希）。 A high score points to AI-generated. Strongest available evidence; measured on the whole image (tiled windows). It locates nothing — never restate its score as a physical artefact or an area."}
```

### 第 3 轮 · user
```
[System: 本会话状态 —— 已执行的专家：
  - probe_dino_expert（E-c8ae152c7f）：aigi_discriminator_score=0.74777（该值的方向：AI-generated）
尚可调用：npr_expert, univfd_expert。每个专家每会话只对整图测量一次：重复请求的返回值与上面已有的 token 相同，系统会跳过执行并白费一轮。请基于上面已给出的结论继续，或直接给出结案。]
```

### 第 4 轮 · gpt
```
<reasoning>
【物理-语义一致性校验】底层 probe_dino 专家反馈强异常（strength=0.75，判定: AI-generated）。
物理现象: 整图判别分数 0.7478（外部判别器的原始输出，未经本地校准）。输入预处理：mock（确定性哈希）。 A high score points to AI-generated. Strongest available evidence; m。
这与视觉层的观察在因果链上吻合。
已考虑图像后处理（如社交媒体压缩）可能对物理指纹造成的衰减影响。
</reasoning>
<call_freq>[550, 550, 950, 950]</call_freq>

```

### 第 5 轮 · user
```
[System: 本次会话不提供 freq。可用工具：npr, probe, univfd。请从可用工具中选择，或直接给出结论。]
```

### 第 6 轮 · gpt
```
<reasoning>
【物理-语义一致性校验】底层 probe_dino 专家反馈强异常（strength=0.75，判定: AI-generated）。
物理现象: 整图判别分数 0.7478（外部判别器的原始输出，未经本地校准）。输入预处理：mock（确定性哈希）。 A high score points to AI-generated. Strongest available evidence; m。
这与视觉层的观察在因果链上吻合。
已考虑图像后处理（如社交媒体压缩）可能对物理指纹造成的衰减影响。
</reasoning>
<verdict>
{"verdict": "Fake", "confidence": 0.9206, "primary_evidence": ["probe", "freq"], "report": "经多轮法证分析，图像在频域和噪声层面展现出典型的AI生成特征。微观物理痕迹与视觉层面的异常在因果链上高度吻合，综合判定为伪造图像。"}
</verdict>

```

## 3. 逐个数值的包（同一专家的不同取值）

### probe 高分（0.93）
```json
{
  "evidence_id": "E-dump-probe_dino_expert",
  "expert": "probe_dino_expert",
  "measurement": "aigi_discriminator_score",
  "status": "measured",
  "scope": "global",
  "score": {
    "name": "aigi_discriminator_score",
    "value": 0.93,
    "range": [
      0.0,
      1.0
    ],
    "direction": "higher_is_fake",
    "is_calibrated_probability": false,
    "threshold": 0.5,
    "threshold_chosen_on": "外部默认",
    "direction_for_this_value": "AI-generated"
  },
  "reading": "整图判别分数 0.9300（外部判别器的原始输出，未经本地校准）。输入预处理：整图 resize(224,224)。 A high score points to AI-generated. Strongest available evidence; measured on the whole image (tiled windows). It locates nothing — never restate its score as a physical artefact or an area."
}
```

### probe 低分（0.02）
```json
{
  "evidence_id": "E-dump-probe_dino_expert",
  "expert": "probe_dino_expert",
  "measurement": "aigi_discriminator_score",
  "status": "measured",
  "scope": "global",
  "score": {
    "name": "aigi_discriminator_score",
    "value": 0.02,
    "range": [
      0.0,
      1.0
    ],
    "direction": "higher_is_fake",
    "is_calibrated_probability": false,
    "threshold": 0.5,
    "threshold_chosen_on": "外部默认",
    "direction_for_this_value": "Real"
  },
  "reading": "整图判别分数 0.0200（外部判别器的原始输出，未经本地校准）。输入预处理：整图 resize(224,224)。 A low score points to Real; this expert holds across formats. Strongest available evidence; measured on the whole image (tiled windows). It locates nothing — never restate its score as a physical artefact or an area."
}
```

### npr 高分（0.98）
```json
{
  "evidence_id": "E-dump-npr_expert",
  "expert": "npr_expert",
  "measurement": "npr_upsampling_artefact_score",
  "status": "measured",
  "scope": "global",
  "score": {
    "name": "npr_upsampling_artefact_score",
    "value": 0.98,
    "range": [
      0.0,
      1.0
    ],
    "direction": "higher_is_fake",
    "is_calibrated_probability": false,
    "threshold": 0.5,
    "threshold_chosen_on": "官方默认",
    "direction_for_this_value": "AI-generated"
  },
  "reading": "整图判别分数 0.9800（外部判别器的原始输出，未经本地校准）。输入预处理：整图 resize(224,224)。 A high score points to AI-generated. Validated under this collection's PNG encoding. Under JPEG its scores are near zero: that only means it did not see the up-sampling artefacts it looks for — a low score there is NOT evidence of a real photograph."
}
```

### npr 低分（0.02）——JPEG 边界
```json
{
  "evidence_id": "E-dump-npr_expert",
  "expert": "npr_expert",
  "measurement": "npr_upsampling_artefact_score",
  "status": "measured",
  "scope": "global",
  "score": {
    "name": "npr_upsampling_artefact_score",
    "value": 0.02,
    "range": [
      0.0,
      1.0
    ],
    "direction": "higher_is_fake",
    "is_calibrated_probability": false,
    "threshold": 0.5,
    "threshold_chosen_on": "官方默认",
    "direction_for_this_value": "Uncertain"
  },
  "reading": "整图判别分数 0.0200（外部判别器的原始输出，未经本地校准）。输入预处理：整图 resize(224,224)。 A low score means this expert did not find the up-sampling artefacts it looks for; **it does not indicate a real photograph.** Validated under this collection's PNG encoding. Under JPEG its scores are near zero: that only means it did not see the up-sampling artefacts it looks for — a low score there is NOT evidence of a real photograph."
}
```

### univfd（0.88）
```json
{
  "evidence_id": "E-dump-univfd_expert",
  "expert": "univfd_expert",
  "measurement": "univfd_linear_head_score",
  "status": "measured",
  "scope": "global",
  "score": {
    "name": "univfd_linear_head_score",
    "value": 0.88,
    "range": [
      0.0,
      1.0
    ],
    "direction": "higher_is_fake",
    "is_calibrated_probability": false,
    "threshold": 0.5,
    "threshold_chosen_on": "官方默认（acc.txt）",
    "direction_for_this_value": "AI-generated"
  },
  "reading": "整图判别分数 0.8800（外部判别器的原始输出，未经本地校准）。输入预处理：centre crop 224x224。 A high score points to AI-generated. Measures a 224x224 centre crop: it never sees the image edges."
}
```

### npr 执行失败
```json
{
  "evidence_id": "E-dump-npr_expert",
  "expert": "npr_expert",
  "measurement": "npr_upsampling_artefact_score",
  "status": "failed",
  "reading": "本次测量执行失败，**没有任何数值**。这不是中性结果，也不是证据——请改用别的工具或直接给出结论。",
  "reason": "RuntimeError: 权重文件不存在"
}
```

## 4. 内部标记扫描
| 文本 | 命中的内部标记 |
| --- | --- |
| 系统提示词 | （无） |
| 会话第 0 轮（user） | （无） |
| 会话第 1 轮（gpt） | （无） |
| 会话第 2 轮（user） | （无） |
| 会话第 3 轮（user） | （无） |
| 会话第 4 轮（gpt） | （无） |
| 会话第 5 轮（user） | （无） |
| 会话第 6 轮（gpt） | （无） |
| 包：probe 高分（0.93） | （无） |
| 包：probe 低分（0.02） | （无） |
| 包：npr 高分（0.98） | （无） |
| 包：npr 低分（0.02）——JPEG 边界 | （无） |
| 包：univfd（0.88） | （无） |
| 包：npr 执行失败 | （无） |

合计命中：**0**（0 才算通过；判定用的是词表，不是我的印象）

包里出现过的字段：`['evidence_id', 'expert', 'measurement', 'reading', 'reason', 'scope', 'score', 'status']`
