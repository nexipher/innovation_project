# 30 图修复前后对照

修复：无 bbox 的 TruFor 三图改为 high-response 包围盒；覆盖纠正要求未测 TruFor/PROBE 先补测，禁止两位专家已测时重复调用；局部仅候选时明确整图 AI-generated 仍保持总体 Fake；整图 AI-generated 且 TruFor 低分时局部改 Not applicable。

## 总表

| 指标 | 原版 before | 原版 after | Δ |
|---|---:|---:|---:|
| 局部篡改检出（/10） | 2 | 3 | +1 |
| 真图局部误报（/10） | 0 | 0 | +0 |
| 真图整图误报（/10） | 2 | 2 | +0 |
| 真图总体误报（/10） | 2 | 2 | +0 |
| 两分项同时判对（/30） | 8 | 12 | +4 |
| 其中篡改图（/10） | 1 | 0 | -1 |
| 其中生成图（/10） | 0 | 6 | +6 |
| 其中真图（/10） | 7 | 6 | -1 |
| 弃权（/30） | 0 | 3 | +3 |
| 专家调用总数 | 54 | 58 | +4 |
| 分项/总体一致（/30） | 30 | 29 | -1 |
| 生成图 local Not applicable（/10） | 0 | 6 | +6 |
| 覆盖检查 failed（/30） | 3 | 1 | -2 |

| 指标 | LoRA before | LoRA after | Δ |
|---|---:|---:|---:|
| 局部篡改检出（/10） | 1 | 4 | +3 |
| 真图局部误报（/10） | 0 | 0 | +0 |
| 真图整图误报（/10） | 2 | 2 | +0 |
| 真图总体误报（/10） | 2 | 2 | +0 |
| 两分项同时判对（/30） | 7 | 14 | +7 |
| 其中篡改图（/10） | 1 | 1 | +0 |
| 其中生成图（/10） | 0 | 6 | +6 |
| 其中真图（/10） | 6 | 7 | +1 |
| 弃权（/30） | 2 | 1 | -1 |
| 专家调用总数 | 49 | 56 | +7 |
| 分项/总体一致（/30） | 28 | 30 | +2 |
| 生成图 local Not applicable（/10） | 0 | 6 | +6 |
| 覆盖检查 failed（/30） | 3 | 2 | -1 |

## 重点案例

| sample | truth whole/local | base before | base after | LoRA before | LoRA after |
|---|---|---|---|---|---|
| `val__autosplice__40489__2__q95` | Real/Tampered | AI-generated/Not tampered/Fake | AI-generated/Not applicable/Fake | AI-generated/Not tampered/Fake | AI-generated/Not applicable/Fake |
| `val__autosplice__41993__0__q95` | Real/Tampered | AI-generated/Uncertain/Fake | Uncertain/Uncertain/Uncertain | AI-generated/Uncertain/Fake | AI-generated/Uncertain/Fake |
| `val__autosplice__42958__1__q95` | Real/Tampered | Real/Tampered/Fake | Real/Uncertain/Fake | Real/Tampered/Fake | Real/Tampered/Fake |
| `val__autosplice__43686__1__q95` | Real/Tampered | AI-generated/Uncertain/Fake | Uncertain/Tampered/Fake | AI-generated/Uncertain/Fake | AI-generated/Tampered/Fake |
| `val__autosplice__56086__0__q95` | Real/Tampered | AI-generated/Not tampered/Fake | AI-generated/Not tampered/Fake | AI-generated/Uncertain/Fake | AI-generated/Uncertain/Fake |
| `val__autosplice__57143__0__q95` | Real/Tampered | AI-generated/Tampered/Fake | AI-generated/Tampered/Fake | AI-generated/Uncertain/Uncertain | AI-generated/Tampered/Fake |
| `val__autosplice__57276__0__q95` | Real/Tampered | Real/Not tampered/Real | Real/Not tampered/Real | Real/Not tampered/Real | Real/Not tampered/Real |
| `val__autosplice__60788__0__q95` | Real/Tampered | AI-generated/Uncertain/Fake | AI-generated/Not applicable/Fake | AI-generated/Uncertain/Fake | AI-generated/Not applicable/Fake |
| `val__autosplice__67065__0__q95` | Real/Tampered | Real/Uncertain/Real | Real/Uncertain/Real | Real/Uncertain/Real | Real/Uncertain/Real |
| `val__autosplice__73275__1__q95` | Real/Tampered | AI-generated/Uncertain/Fake | AI-generated/Tampered/Fake | AI-generated/Uncertain/Fake | Uncertain/Tampered/Fake |
| `val__genimage__BigGAN__739_biggan_00039__q95` | AI-generated/Not applicable | AI-generated/Not tampered/Fake | AI-generated/Not applicable/Fake | AI-generated/Uncertain/Fake | AI-generated/Not applicable/Fake |
| `val__genimage__Real__d3429f29-5d3a-4da3-829a-0fb387554a98__q95` | Real/Not tampered | Real/Not tampered/Real | Real/Not tampered/Real | Real/Uncertain/Real | Real/Not tampered/Real |
| `val__genimage__VQDM__VQDM_1000_200_01_143_vqdm_00143__q95` | AI-generated/Not applicable | AI-generated/Uncertain/Fake | AI-generated/Not applicable/Fake | AI-generated/Not tampered/Fake | AI-generated/Not applicable/Fake |

## 边界

- 同一 30 图、同一 greedy/non-thinking 设置；修复只改证据呈现和覆盖纠正，未重训、未改专家权重/阈值。
- `consistent` 只检查总体与分项的标签关系（Fake 分项是否存在），不等同于真值正确。
- 覆盖检查失败是记录，不会静默改写模型标签。
