# 联合取证规则更新复验：历史口径 vs 独立局部口径

新规则：整图生成与局部编辑独立；局部只输出 Tampered / Not tampered / Uncertain，取消 Not applicable；whole=AI-generated 时总体必须 Fake。历史 `validation_fixed/base` 标签未改，仅在对照中作为旧口径。

## 总表

| 指标 | 旧 base | 新 base | Δ |
|---|---:|---:|---:|
| 局部篡改检出（10 伪造） | 3 | 2 | -1 |
| 真图局部误报（10 真图） | 0 | 0 | +0 |
| 生成图局部误报（10 构造确认负例） | 2 | 2 | +0 |
| 真图局部 Uncertain | 2 | 3 | +1 |
| 生成图局部 Uncertain | 2 | 1 | -1 |
| 局部 Uncertain 总数（30） | 7 | 10 | +3 |
| 总体弃权（30） | 3 | 0 | -3 |
| 分项/总体结案一致（30） | 29 | 29 | +0 |
| 总体与真值一致（30） | 23 | 26 | +3 |
| 专家调用总数 | 58 | 59 | +1 |

## 重点案例

| sample | truth | 旧 whole/local/overall | 新 whole/local/overall | 新一致性 |
|---|---|---|---|---|
| `val__autosplice__40489__2__q95` | Real/Tampered | AI-generated/Not applicable/Fake | AI-generated/Uncertain/Fake | True |
| `val__autosplice__41993__0__q95` | Real/Tampered | Uncertain/Uncertain/Uncertain | AI-generated/Uncertain/Fake | True |
| `val__autosplice__42958__1__q95` | Real/Tampered | Real/Uncertain/Fake | Real/Uncertain/Fake | False |
| `val__autosplice__57143__0__q95` | Real/Tampered | AI-generated/Tampered/Fake | AI-generated/Uncertain/Fake | True |
| `val__autosplice__60788__0__q95` | Real/Tampered | AI-generated/Not applicable/Fake | AI-generated/Not tampered/Fake | True |
| `val__autosplice__67065__0__q95` | Real/Tampered | Real/Uncertain/Real | Real/Tampered/Fake | True |

## 读法

- 新口径把生成图也视为可能被后续局部篡改；旧 `Not applicable` 仅作为历史记录保留，不计入新规则的局部负例。
- 生成图局部负例的构造依据见 `truth_provenance.csv`：GenImage manifest 记录 generator-only q95 派生，没有局部编辑步骤或 mask。
- 旧 base 的 `Not applicable` 在对照表中原样展示，不重写历史文件。

## 边界

本轮只改提示/结案检查和复验；裁剪、专家、Qwen 配置固定；不重训、不扩样、不碰 test。
